"""
WebSocket Server for Echo Voice Support.
Handles real-time voice communication with clients.
"""

import asyncio
import json
import base64
import uuid
from typing import Set
import structlog

from voice_engine import VoiceEngine
from config.settings import config

logger = structlog.get_logger()


class VoiceWebSocketServer:
    """WebSocket server for voice interactions."""

    def __init__(self, host: str = None, port: int = None):
        self.host = host or config.websocket.host
        self.port = port or config.websocket.port
        self.clients: Set[asyncio.Queue] = set()
        self.voice_engine = VoiceEngine()
        self._server = None
        self._running = False

    async def start(self):
        """Start the WebSocket server."""
        await self.voice_engine.start()

        self._server = await asyncio.start_server(
            self._handle_client, self.host, self.port
        )

        self._running = True
        logger.info("WebSocket server started", host=self.host, port=self.port)

        async with self._server:
            await self._server.serve_forever()

    async def stop(self):
        """Stop the WebSocket server."""
        self._running = False

        if self._server:
            self._server.close()
            await self._server.wait_closed()

        await self.voice_engine.stop()
        logger.info("WebSocket server stopped")

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ):
        """Handle a client connection."""
        client_id = str(uuid.uuid4())
        session_id = None
        message_queue = asyncio.Queue()

        self.clients.add(message_queue)

        peername = writer.get_extra_info("peername")
        logger.info("Client connected", client_id=client_id, address=peername)

        try:
            await self._send_message(
                writer, {"type": "connected", "client_id": client_id}
            )

            session_id = await self.voice_engine.start_session(client_id)

            response_task = asyncio.create_task(
                self._process_response_stream(writer, message_queue)
            )

            reader_task = asyncio.create_task(
                self._read_client_messages(reader, message_queue, client_id)
            )

            await asyncio.gather(reader_task, response_task, return_exceptions=True)

        except Exception as e:
            logger.error("Client handler error", client_id=client_id, error=str(e))

        finally:
            if session_id:
                await self.voice_engine.end_session(session_id)

            self.clients.discard(message_queue)
            writer.close()
            await writer.wait_closed()

            logger.info("Client disconnected", client_id=client_id)

    async def _read_client_messages(
        self, reader: asyncio.StreamReader, queue: asyncio.Queue, client_id: str
    ):
        """Read messages from client."""
        try:
            while self._running:
                try:
                    data = await asyncio.wait_for(reader.read(4096), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                if not data:
                    break

                try:
                    message = json.loads(data.decode())
                    message["client_id"] = client_id
                    await queue.put(message)

                except json.JSONDecodeError:
                    logger.warning("Invalid JSON received", client_id=client_id)

        except Exception as e:
            logger.error("Read error", client_id=client_id, error=str(e))

    async def _process_response_stream(
        self, writer: asyncio.StreamWriter, queue: asyncio.Queue
    ):
        """Process and respond to client messages."""
        try:
            while self._running:
                try:
                    message = await asyncio.wait_for(queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue

                msg_type = message.get("type")
                client_id = message.get("client_id")

                if msg_type == "audio":
                    audio_data = base64.b64decode(message.get("data", ""))

                    result = await self.voice_engine.process_voice_input(
                        audio_data=audio_data, user_id=client_id
                    )

                    if result.audio_output:
                        response = {
                            "type": "response_audio",
                            "data": base64.b64encode(result.audio_output).decode(),
                            "transcript": result.text_input,
                            "response": result.text_output,
                            "timing": {
                                "stt_ms": result.timing.stt_time_ms
                                if result.timing
                                else 0,
                                "llm_ms": result.timing.llm_time_ms
                                if result.timing
                                else 0,
                                "tts_ms": result.timing.tts_time_ms
                                if result.timing
                                else 0,
                                "total_ms": result.timing.total_time_ms
                                if result.timing
                                else 0,
                            },
                        }
                        await self._send_message(writer, response)

                elif msg_type == "text":
                    text_input = message.get("text", "")

                    result = await self.voice_engine.process_text_input(
                        text_input=text_input, user_id=client_id
                    )

                    if result.audio_output:
                        response = {
                            "type": "response_audio",
                            "data": base64.b64encode(result.audio_output).decode(),
                            "response": result.text_output,
                            "timing": {
                                "llm_ms": result.timing.llm_time_ms
                                if result.timing
                                else 0,
                                "tts_ms": result.timing.tts_time_ms
                                if result.timing
                                else 0,
                                "total_ms": result.timing.total_time_ms
                                if result.timing
                                else 0,
                            },
                        }
                        await self._send_message(writer, response)

                elif msg_type == "ping":
                    await self._send_message(writer, {"type": "pong"})

                elif msg_type == "status":
                    status = {
                        "type": "status",
                        "connected": True,
                        "llm_available": await self.voice_engine.llm.check_health(),
                        "clients_connected": len(self.clients),
                    }
                    await self._send_message(writer, status)

        except Exception as e:
            logger.error("Response stream error", error=str(e))

    async def _send_message(self, writer: asyncio.StreamWriter, message: dict):
        """Send a message to the client."""
        try:
            data = json.dumps(message).encode()
            writer.write(data + b"\n")
            await writer.drain()
        except Exception as e:
            logger.error("Send error", error=str(e))


async def main():
    """Main entry point."""
    import signal

    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ]
    )

    server = VoiceWebSocketServer()

    loop = asyncio.get_event_loop()

    def shutdown_handler():
        logger.info("Shutdown signal received")
        asyncio.create_task(server.stop())

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, shutdown_handler)

    try:
        await server.start()
    except KeyboardInterrupt:
        await server.stop()


if __name__ == "__main__":
    asyncio.run(main())
