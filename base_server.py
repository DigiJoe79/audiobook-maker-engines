"""
Base Engine Server - Abstract FastAPI Server for All Engine Types

All engine servers inherit from this class and only need to implement:
- load_model(model_name: str)
- get_available_models() -> List[ModelInfo]
- unload_model()

TTS engines should use BaseTTSServer (from base_tts_server.py)
STT and Audio Analysis engines should use BaseQualityServer (from base_quality_server.py)
Text Processing engines should use BaseTextServer (from base_text_server.py)
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from typing import Any, Dict, List, Optional
from pathlib import Path
from abc import ABC, abstractmethod
import uvicorn
from loguru import logger
import traceback
import sys
import signal
import asyncio
import os
import yaml


# ============= Configure Loguru (same format as main backend) =============
# Remove default handler and configure to match main.py format (no date)
# Log level can be set via LOG_LEVEL environment variable (DEBUG, INFO, WARNING, ERROR)
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    colorize=True
)


# ============= CamelCase Conversion Helper =============
# NOTE: This is intentionally duplicated from backend/models/response_models.py
# because engine servers run in isolated VENVs without access to backend modules.
# Each engine must be self-contained with its own copy of shared utilities.

def to_camel(string: str) -> str:
    """
    Convert snake_case string to camelCase.

    Examples:
        engine_model_name → engineModelName
        current_engine_model → currentEngineModel
    """
    components = string.split('_')
    return components[0] + ''.join(x.title() for x in components[1:])


class CamelCaseModel(BaseModel):
    """
    Base model with automatic snake_case to camelCase conversion.

    All engine server models inherit from this to ensure consistent
    API response formatting (Python snake_case → JSON camelCase).
    """
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,  # Accept both snake_case and camelCase input
    )


# ============= Request/Response Models =============

class LoadRequest(CamelCaseModel):
    """Request to load a specific model"""
    engine_model_name: str


class LoadResponse(CamelCaseModel):
    """Response after loading model"""
    status: str  # "loaded", "error"
    engine_model_name: Optional[str] = None
    error: Optional[str] = None


class HealthResponse(CamelCaseModel):
    """Health check response"""
    status: str  # "ready", "loading", "processing", "error"
    engine_model_loaded: bool
    current_engine_model: Optional[str] = None
    device: str = "cpu"  # "cpu" or "cuda"
    error: Optional[str] = None
    package_version: Optional[str] = None  # Dynamic version from pip package
    gpu_memory_used_mb: Optional[int] = None  # GPU memory currently used (MB)
    gpu_memory_total_mb: Optional[int] = None  # GPU memory total (MB)


class ShutdownResponse(CamelCaseModel):
    """Shutdown acknowledgment"""
    status: str  # "shutting_down"


# ============= Model Info Types =============

class ModelField(CamelCaseModel):
    """Dynamic metadata field for a model"""
    key: str          # e.g., "size_mb", "speed", "accuracy"
    value: Any        # e.g., 39, "~10x realtime", "lowest"
    field_type: str   # "number", "string", "percent" (using field_type to avoid Pydantic conflict)


class ModelInfo(CamelCaseModel):
    """Model information with dynamic metadata"""
    name: str                           # e.g., "tiny", "base", "multilingual"
    display_name: str                   # e.g., "Tiny (39 MB)", "Multilingual (Pretrained)"
    languages: List[str] = []           # ISO language codes this model supports (e.g., ["de", "en"])
    fields: List[ModelField] = []       # Dynamic metadata (optional)


class ModelsResponse(CamelCaseModel):
    """Response for /models endpoint"""
    models: List[ModelInfo]
    default_model: Optional[str] = None
    device: str = "cpu"


# ============= Engine Info Types (for /info endpoint) =============

class UpstreamInfo(CamelCaseModel):
    """Upstream project attribution"""
    name: str                           # e.g., "Coqui TTS"
    url: str                            # e.g., "https://github.com/coqui-ai/TTS"
    license: str                        # e.g., "MPL-2.0"


class ParameterSchema(CamelCaseModel):
    """Parameter definition for UI rendering"""
    type: str                           # "float", "int", "bool", "string"
    label: str                          # i18n key for label
    description: str                    # i18n key for description
    default: Any                        # Default value
    min: Optional[float] = None         # Minimum value (for numeric types)
    max: Optional[float] = None         # Maximum value (for numeric types)
    step: Optional[float] = None        # Step size (for numeric types)
    readonly: bool = False              # If true, parameter cannot be changed
    category: Optional[str] = None      # UI grouping category


class EngineModelConfig(CamelCaseModel):
    """Model configuration from engine.yaml"""
    name: str                           # Model identifier
    display_name: str                   # Human-readable name
    # Additional fields are dynamic (size_mb, vram_gb, etc.)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EngineInfoResponse(CamelCaseModel):
    """
    Static engine metadata from engine.yaml.

    Used for Docker discovery - container exposes this via /info endpoint
    so the app can register custom engines without manual configuration.
    """
    # Identity
    name: str                                       # Engine identifier
    display_name: str                               # Human-readable name
    engine_type: str                                # "tts", "stt", "text", "audio"
    description: Optional[str] = None               # Engine description

    # Upstream attribution
    upstream: Optional[UpstreamInfo] = None

    # Languages
    supported_languages: List[str] = Field(default_factory=list)

    # Constraints
    constraints: Dict[str, Any] = Field(default_factory=dict)

    # Capabilities
    capabilities: Dict[str, Any] = Field(default_factory=dict)

    # Parameters schema for UI
    parameters: Dict[str, ParameterSchema] = Field(default_factory=dict)

    # Models
    models: List[EngineModelConfig] = Field(default_factory=list)
    default_model: Optional[str] = None
    auto_discover_models: bool = False

    # Installation info
    requires_gpu: bool = False

    # Schema version
    schema_version: int = 2


# ============= Base Engine Server =============

class BaseEngineServer(ABC):
    """
    Abstract base class for all engine servers (TTS, STT, Text, Audio)

    Engines need to implement these methods:
    - load_model(model_name) - Load a model into memory
    - get_available_models() - Return list of available models
    - unload_model() - Unload model and free resources

    TTS engines should use BaseTTSServer which adds the /generate endpoint.

    All FastAPI routes, error handling, and lifecycle management are handled here.
    """

    def __init__(self, engine_name: str, display_name: str, config_path: Optional[str] = None):
        """
        Initialize engine server

        Args:
            engine_name: Engine identifier (e.g., "xtts", "whisper", "spacy")
            display_name: Human-readable name (e.g., "XTTS v2", "Whisper STT")
            config_path: Optional path to engine.yaml (for /info endpoint)
        """
        self.engine_name = engine_name
        self.display_name = display_name
        self.app = FastAPI(title=f"{display_name} Server")

        # State
        self.status = "ready"  # ready, loading, processing, error
        self.model_loaded = False
        self.current_model = None
        self.default_model: Optional[str] = None  # Subclasses can set this
        self.error_message = None
        self.shutdown_requested = False
        self.server: Optional[uvicorn.Server] = None  # Server reference for graceful shutdown
        self.device = "cpu"  # Default device, subclasses can override (e.g., "cuda")

        # Determine base path (Docker: /app, Subprocess: engine directory)
        if Path("/app").exists() and Path("/app/server.py").exists():
            self.base_path = Path("/app")
        else:
            # Subprocess: use the directory where server.py is located
            self.base_path = Path(sys.argv[0]).parent.resolve() if sys.argv[0] else Path(__file__).parent

        # Model directories (engines use these paths)
        self.models_dir = self.base_path / "models"
        self.external_models_dir = self.base_path / "external_models"

        # Ensure models directory exists
        self.models_dir.mkdir(parents=True, exist_ok=True)

        logger.debug(f"[{self.engine_name}] Base path: {self.base_path}")
        logger.debug(f"[{self.engine_name}] Models dir: {self.models_dir}")

        # Engine config (loaded from engine.yaml for /info endpoint)
        self._engine_config: Dict[str, Any] = {}
        if config_path:
            self._load_engine_config(config_path)
        else:
            # Auto-detect engine.yaml in base_path
            auto_config = self.base_path / "engine.yaml"
            if auto_config.exists():
                self._load_engine_config(str(auto_config))

        # Thread safety for model operations
        self._model_lock = asyncio.Lock()

        # Setup routes
        self._setup_routes()

        logger.debug(f"[{self.engine_name}] BaseEngineServer initialized")

    def _setup_routes(self):
        """Setup FastAPI routes (called automatically)"""

        @self.app.post("/load", response_model=LoadResponse)
        async def load_endpoint(request: LoadRequest):
            """Load a specific model into memory"""
            async with self._model_lock:
                try:
                    self.status = "loading"
                    self.error_message = None

                    # Call engine-specific implementation
                    self.load_model(request.engine_model_name)

                    self.model_loaded = True
                    self.current_model = request.engine_model_name
                    self.status = "ready"

                    return LoadResponse(status="loaded", engine_model_name=request.engine_model_name)

                except HTTPException:
                    raise
                except Exception as e:
                    self.status = "error"
                    self.error_message = str(e)
                    logger.error(f"[{self.engine_name}] Model loading failed: {e}")
                    logger.error(traceback.format_exc())
                    raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/models", response_model=ModelsResponse)
        async def models_endpoint():
            """Return available models for this engine"""
            try:
                models = self.get_available_models()
                return ModelsResponse(
                    models=models,
                    default_model=self.default_model,
                    device=self.device
                )
            except Exception as e:
                logger.error(f"[{self.engine_name}] Failed to get models: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/health", response_model=HealthResponse)
        async def health_endpoint():
            """Health check"""
            gpu_used, gpu_total = self._get_gpu_memory()
            return HealthResponse(
                status=self.status,
                engine_model_loaded=self.model_loaded,
                current_engine_model=self.current_model,
                device=self.device,
                error=self.error_message,
                package_version=self.get_package_version(),
                gpu_memory_used_mb=gpu_used,
                gpu_memory_total_mb=gpu_total
            )

        @self.app.get("/info", response_model=EngineInfoResponse)
        async def info_endpoint():
            """
            Static engine metadata from engine.yaml.

            Used for Docker discovery - the app queries this endpoint to get
            engine configuration without requiring manual input.
            """
            return self._build_engine_info()

        @self.app.post("/shutdown", response_model=ShutdownResponse)
        async def shutdown_endpoint():
            """Graceful shutdown request"""
            self.shutdown_requested = True

            # Unload model to free resources
            async with self._model_lock:
                try:
                    self.unload_model()
                except Exception as e:
                    logger.error(f"[{self.engine_name}] Error during unload: {e}")

            # Schedule server shutdown after response is sent (100ms delay)
            if self.server:
                asyncio.create_task(self._delayed_shutdown())

            return ShutdownResponse(status="shutting_down")

        async def _delayed_shutdown_impl():
            """Internal helper: shutdown server after brief delay"""
            await asyncio.sleep(0.1)  # Brief delay to let response be sent
            if self.server:
                self.server.should_exit = True

        # Store as instance method so shutdown endpoint can access it
        self._delayed_shutdown = _delayed_shutdown_impl

    # ============= Abstract Methods (Engine-Specific) =============

    @abstractmethod
    def load_model(self, model_name: str) -> None:
        """
        Load model into memory (engine-specific)

        Args:
            model_name: Model identifier (e.g., "v2.0.3", "base", "de_core_news_sm")

        Raises:
            Exception: If loading fails
        """
        pass

    @abstractmethod
    def get_available_models(self) -> List[ModelInfo]:
        """
        Return list of available models for this engine.

        Each engine implements this to return its available models.
        For engines without ML models, return a single
        "default" model entry.

        Returns:
            List of ModelInfo objects describing available models
        """
        pass

    @abstractmethod
    def unload_model(self) -> None:
        """
        Unload model and free resources (engine-specific)
        """
        pass

    def get_package_version(self) -> Optional[str]:
        """
        Return package version for health endpoint (optional override).

        Subclasses can override this to return their pip package version.
        Default returns None.

        Returns:
            Version string (e.g., "5.1.0") or None
        """
        return None

    # ============= Engine Config (for /info endpoint) =============

    def _load_engine_config(self, config_path: str) -> None:
        """
        Load engine configuration from engine.yaml file.

        Args:
            config_path: Path to engine.yaml file
        """
        try:
            path = Path(config_path)
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    self._engine_config = yaml.safe_load(f) or {}
                logger.debug(f"[{self.engine_name}] Loaded engine config from {config_path}")
            else:
                logger.warning(f"[{self.engine_name}] Config file not found: {config_path}")
        except Exception as e:
            logger.error(f"[{self.engine_name}] Failed to load engine config: {e}")
            self._engine_config = {}

    def _build_engine_info(self) -> EngineInfoResponse:
        """
        Build EngineInfoResponse from loaded config.

        Returns:
            EngineInfoResponse with all engine metadata
        """
        config = self._engine_config

        # Parse upstream info
        upstream = None
        if 'upstream' in config:
            up = config['upstream']
            upstream = UpstreamInfo(
                name=up.get('name', ''),
                url=up.get('url', ''),
                license=up.get('license', '')
            )

        # Parse parameters schema
        parameters: Dict[str, ParameterSchema] = {}
        if 'parameters' in config:
            for param_name, param_config in config['parameters'].items():
                if isinstance(param_config, dict):
                    parameters[param_name] = ParameterSchema(
                        type=param_config.get('type', 'string'),
                        label=param_config.get('label', param_name),
                        description=param_config.get('description', ''),
                        default=param_config.get('default'),
                        min=param_config.get('min'),
                        max=param_config.get('max'),
                        step=param_config.get('step'),
                        readonly=param_config.get('readonly', False),
                        category=param_config.get('category')
                    )

        # Parse models
        models: List[EngineModelConfig] = []
        if 'models' in config:
            for model in config['models']:
                if isinstance(model, dict):
                    # Extract standard fields, put rest in metadata
                    name = model.get('name', '')
                    display_name = model.get('display_name', name)
                    metadata = {k: v for k, v in model.items()
                               if k not in ('name', 'display_name')}
                    models.append(EngineModelConfig(
                        name=name,
                        display_name=display_name,
                        metadata=metadata
                    ))

        # Determine requires_gpu from installation or variants
        requires_gpu = False
        if 'installation' in config:
            requires_gpu = config['installation'].get('requires_gpu', False)
        elif 'variants' in config:
            # Check if any variant requires GPU
            for variant in config.get('variants', []):
                if variant.get('requires_gpu', False):
                    requires_gpu = True
                    break

        return EngineInfoResponse(
            name=config.get('name', self.engine_name),
            display_name=config.get('display_name', self.display_name),
            engine_type=config.get('engine_type', 'unknown'),
            description=config.get('description'),
            upstream=upstream,
            supported_languages=config.get('supported_languages', []),
            constraints=config.get('constraints', {}),
            capabilities=config.get('capabilities', {}),
            parameters=parameters,
            models=models,
            default_model=config.get('default_model', self.default_model),
            auto_discover_models=config.get('auto_discover_models', False),
            requires_gpu=requires_gpu,
            schema_version=config.get('schema_version', 2)
        )

    def _get_gpu_memory(self) -> tuple[Optional[int], Optional[int]]:
        """
        Get GPU memory usage if CUDA is available.

        Returns:
            Tuple of (used_mb, total_mb) or (None, None) if not on CUDA
        """
        if self.device != "cuda":
            return None, None

        try:
            import torch
            if torch.cuda.is_available():
                # Get memory for current device
                used = torch.cuda.memory_allocated() // (1024 * 1024)
                total = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
                return used, total
        except Exception:
            pass

        return None, None

    # ============= Server Lifecycle =============

    def run(self, port: int, host: str = "127.0.0.1"):
        """
        Start the FastAPI server

        Args:
            port: Port to listen on
            host: Host to bind to (default: localhost only)
        """
        # Create uvicorn server manually to enable graceful shutdown
        config = uvicorn.Config(
            self.app,
            host=host,
            port=port,
            log_level="error",  # Only show errors, suppress INFO logs
            access_log=False     # Disable access logs (we log in endpoints)
        )
        self.server = uvicorn.Server(config)

        # Custom signal handler for clean shutdown without tracebacks
        def handle_exit_signal(signum, frame):
            """Handle SIGINT/SIGTERM for clean shutdown"""
            if self.server:
                self.server.should_exit = True

        # Install signal handlers
        signal.signal(signal.SIGINT, handle_exit_signal)
        signal.signal(signal.SIGTERM, handle_exit_signal)

        # Run server (blocks until shutdown via /shutdown endpoint or signal)
        try:
            asyncio.run(self.server.serve())
        except (KeyboardInterrupt, SystemExit, asyncio.CancelledError):
            pass  # Clean exit, no traceback
