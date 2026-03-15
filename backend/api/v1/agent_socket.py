from fastapi import APIRouter, WebSocket
from dishka.integrations.fastapi import FromDishka
from api.v1.agent_socket_controller import AgentSocketController

ws_router: APIRouter = APIRouter()

@ws_router.websocket('/agent-session')
async def agent_session(
    websocket: WebSocket,
    controller: FromDishka[AgentSocketController],
) -> None:
    """
    WebSocket endpoint delegating to the AgentSocketController.
    """
    await controller.handle_session(websocket)
