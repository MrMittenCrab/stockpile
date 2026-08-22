"""FastAPI application factory for local fixed-seat Stockpile Lite games."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Header, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .. import stockpile_interface as interface
from .schemas import (
    ActionRequestV1,
    ChatRequestV1,
    ChatResponseV1,
    CreateGameRequest,
    CreateGameResponse,
    ErrorBody,
    ErrorResponse,
    GameViewV1,
    IntegerLimits,
    OptionDescriptor,
    SetupDefaults,
    SetupResponse,
)
from .sessions import SessionError, SessionStore
from .v2_schemas import (
    AcknowledgementRequestV2,
    ActionRequestV2,
    CreateGameRequestV2,
    CreateGameResponseV2,
    GameViewV2,
    OptionDescriptorV2,
    SetupResponseV2,
    SupplyRequestV2,
)
from .v2_sessions import V2SessionStore


def _error_payload(code: str, message: str) -> dict[str, Any]:
    return ErrorResponse(error=ErrorBody(code=code, message=message)).model_dump(
        mode="json"
    )


def _bearer_token(authorization: str | None) -> str | None:
    if authorization is None:
        return None
    scheme, separator, token = authorization.partition(" ")
    if not separator or scheme.casefold() != "bearer" or not token.strip():
        return None
    return token.strip()


def create_app(
    store: SessionStore | None = None,
    *,
    store_v2: V2SessionStore | None = None,
) -> FastAPI:
    """Create an isolated application; tests may inject a fresh session store."""

    sessions = store or SessionStore()
    browser_sessions = store_v2 or V2SessionStore()
    app = FastAPI(
        title="Stockpile Lite local play API",
        version="2.0.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.session_store = sessions
    app.state.v2_session_store = browser_sessions

    @app.middleware("http")
    async def prevent_storage(request: Request, call_next):  # type: ignore[no-untyped-def]
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["Pragma"] = "no-cache"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.exception_handler(SessionError)
    async def session_error_handler(
        _request: Request, error: SessionError
    ) -> JSONResponse:
        headers = {"WWW-Authenticate": "Bearer"} if error.status_code == 401 else None
        return JSONResponse(
            status_code=error.status_code,
            content=_error_payload(error.code, str(error)),
            headers=headers,
        )

    @app.exception_handler(RequestValidationError)
    async def request_validation_handler(
        _request: Request, error: RequestValidationError
    ) -> JSONResponse:
        first = error.errors()[0] if error.errors() else {}
        message = str(first.get("msg", "Request validation failed"))
        return JSONResponse(
            status_code=422,
            content=_error_payload("invalid_request", message),
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(
        _request: Request, error: StarletteHTTPException
    ) -> JSONResponse:
        code = "not_found" if error.status_code == 404 else "http_error"
        return JSONResponse(
            status_code=error.status_code,
            content=_error_payload(code, str(error.detail)),
        )

    @app.get("/api/v1/setup", response_model=SetupResponse)
    def get_setup() -> SetupResponse:
        defaults = interface.resolve_configuration(interface.ConfigurationMode.LITE)
        return SetupResponse(
            defaults=SetupDefaults(
                player_count=defaults.player_count,
                round_count=defaults.round_count,
            ),
            player_limits=IntegerLimits(minimum=2, maximum=5),
            round_limits=IntegerLimits(minimum=1, maximum=10),
            options=[
                OptionDescriptor(
                    key="market_impact",
                    label="Market Impact",
                    description="Add Market Impact cards and the Action phase.",
                    default=defaults.impact,
                ),
                OptionDescriptor(
                    key="starting_share",
                    label="Starting Share",
                    description="Deal one starting share to each player.",
                    default=defaults.hand,
                ),
                OptionDescriptor(
                    key="trading_fees",
                    label="Trading Fees",
                    description="Include Trading Fee cards.",
                    default=defaults.fees,
                ),
                OptionDescriptor(
                    key="dividends",
                    label="Dividends",
                    description="Include dividend forecasts and claims.",
                    default=defaults.dividend,
                ),
                OptionDescriptor(
                    key="sell_order",
                    label="Sell Order",
                    description="Make selling sequential and publicly ordered.",
                    default=defaults.sell_order,
                ),
            ],
        )

    @app.post(
        "/api/v1/games",
        response_model=CreateGameResponse,
        status_code=status.HTTP_201_CREATED,
    )
    def create_game(request: CreateGameRequest) -> CreateGameResponse:
        _session, response = sessions.create(request)
        return response

    @app.get("/api/v1/games/{game_id}/view", response_model=GameViewV1)
    def get_view(
        game_id: str, authorization: str | None = Header(default=None)
    ) -> GameViewV1:
        session, player_id = sessions.authenticate(
            game_id, _bearer_token(authorization)
        )
        return sessions.view(session, player_id)

    @app.post("/api/v1/games/{game_id}/actions", response_model=GameViewV1)
    def submit_action(
        game_id: str,
        request: ActionRequestV1,
        authorization: str | None = Header(default=None),
    ) -> GameViewV1:
        session, player_id = sessions.authenticate(
            game_id, _bearer_token(authorization)
        )
        return sessions.act(
            session,
            player_id,
            action_id=request.action_id,
            expected_revision=request.expected_revision,
        )

    @app.post(
        "/api/v1/games/{game_id}/chat",
        response_model=ChatResponseV1,
        status_code=status.HTTP_201_CREATED,
    )
    def submit_chat(
        game_id: str,
        request: ChatRequestV1,
        authorization: str | None = Header(default=None),
    ) -> ChatResponseV1:
        session, player_id = sessions.authenticate(
            game_id, _bearer_token(authorization)
        )
        return ChatResponseV1(
            chat_message=sessions.add_chat(session, player_id, request.message)
        )

    @app.get("/api/v2/setup", response_model=SetupResponseV2)
    def get_setup_v2() -> SetupResponseV2:
        defaults = interface.resolve_configuration(
            interface.ConfigurationMode.LITE,
            player_count=2,
            round_count=6,
        )
        return SetupResponseV2(
            options=[
                OptionDescriptorV2(
                    key="dividends", label="DIVIDEND", default=defaults.dividend
                ),
                OptionDescriptorV2(
                    key="trading_fees", label="FEES", default=defaults.fees
                ),
                OptionDescriptorV2(
                    key="market_impact", label="IMPACT", default=defaults.impact
                ),
                OptionDescriptorV2(
                    key="sell_order", label="SELL ORDER", default=defaults.sell_order
                ),
            ]
        )

    @app.post(
        "/api/v2/games",
        response_model=CreateGameResponseV2,
        status_code=status.HTTP_201_CREATED,
    )
    def create_game_v2(request: CreateGameRequestV2) -> CreateGameResponseV2:
        _session, response = browser_sessions.create(request)
        return response

    @app.get("/api/v2/games/{game_id}/view", response_model=GameViewV2)
    def get_view_v2(
        game_id: str, authorization: str | None = Header(default=None)
    ) -> GameViewV2:
        session = browser_sessions.authenticate(game_id, _bearer_token(authorization))
        return browser_sessions.view(session)

    @app.post("/api/v2/games/{game_id}/actions", response_model=GameViewV2)
    def submit_action_v2(
        game_id: str,
        request: ActionRequestV2,
        authorization: str | None = Header(default=None),
    ) -> GameViewV2:
        session = browser_sessions.authenticate(game_id, _bearer_token(authorization))
        return browser_sessions.act(
            session,
            action_id=request.action_id,
            expected_revision=request.expected_revision,
        )

    @app.post("/api/v2/games/{game_id}/supply", response_model=GameViewV2)
    def submit_supply_v2(
        game_id: str,
        request: SupplyRequestV2,
        authorization: str | None = Header(default=None),
    ) -> GameViewV2:
        session = browser_sessions.authenticate(game_id, _bearer_token(authorization))
        return browser_sessions.supply(
            session,
            plan_id=request.plan_id,
            expected_revision=request.expected_revision,
        )

    @app.post(
        "/api/v2/games/{game_id}/acknowledgements", response_model=GameViewV2
    )
    def acknowledge_v2(
        game_id: str,
        request: AcknowledgementRequestV2,
        authorization: str | None = Header(default=None),
    ) -> GameViewV2:
        session = browser_sessions.authenticate(game_id, _bearer_token(authorization))
        return browser_sessions.acknowledge(
            session,
            checkpoint_id=request.checkpoint_id,
            expected_revision=request.expected_revision,
        )

    return app


app = create_app()
