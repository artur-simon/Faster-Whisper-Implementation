# WispLive - Comprehensive Architecture Analysis

## Current Architecture Overview

WispLive is a real-time audio transcription application that captures audio from a microphone and transcribes it using Whisper models. The application follows a **layered architecture** with some mixing of concerns.

### Architectural Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                           │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  UI Components (MainWindow, ConfigWindow, AudioVisualizer) │ │
│  │  - TranscriptionHandler (UI Logic Coordinator)              │ │
│  │  - ThemeManager, TrayIconManager                            │ │
│  └─────────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                   APPLICATION/BUSINESS LAYER                    │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  TranscriptionController (Use Case Coordinator)            │ │
│  │  - Orchestrates transcription workflow                      │ │
│  │  - Manages TranscriptionOrchestrator                        │ │
│  └─────────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                   DOMAIN/BUSINESS LOGIC LAYER                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  TranscriptionOrchestrator (Core Business Logic)           │ │
│  │  - Audio processing pipeline                                │ │
│  │  - Overlap resolution algorithms                           │ │
│  │  - Local agreement tracking                                │ │
│  │  TranscriptionEngine (ML Model Wrapper)                    │ │
│  │  - Whisper model interface                                 │ │
│  └─────────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                   INFRASTRUCTURE LAYER                          │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  AudioDataProvider (Audio Infrastructure)                  │ │
│  │  - AudioCapture (Hardware Access)                          │ │
│  │  - Buffer management                                       │ │
│  │  DocumentWriter (File I/O)                                 │ │
│  │  ConfigManager (Configuration Persistence)                 │ │
│  │  AppStateManager (Application State)                       │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Detailed Component Analysis

### 🎯 **Core Domain Entities (app/models.py)**
```python
# Domain Models - Pure data structures
@dataclass(frozen=True)
class Word:                    # Atomic transcription unit
    text: str
    start: float
    end: float
    probability: float

@dataclass(frozen=True)
class TranscriptionSegment:    # Whisper output unit
    text: str
    words: List[Word]
    no_speech_prob: float

@dataclass()
class TranscriptionConfig:     # Business configuration
    model_size: str
    device: str
    # ... transcription parameters
```

### 🏗️ **Infrastructure Layer**

#### Audio Subsystem
```
AudioDataProvider (NEW - Clean abstraction)
├── AudioCapture (Hardware access)
│   ├── SoundDevice integration
│   ├── Buffer management
│   └── Audio resampling
└── Multiple consumer support
    ├── TranscriptionOrchestrator
    └── AudioVisualizer
```

#### Persistence & Configuration
```
ConfigManager
├── JSON file I/O
├── Default configuration
└── Runtime config updates

AppStateManager
├── Window geometry
├── UI state persistence
└── Theme preferences

DocumentWriter
├── Text file output
├── Real-time writing
└── Line break management
```

### 🎼 **Domain Layer (Business Logic)**

#### TranscriptionEngine
```
TranscriptionEngine (ML Model Wrapper)
├── WhisperModel integration
├── Audio preprocessing
├── VAD filtering
└── Word-level timestamps
```

#### TranscriptionOrchestrator (Complex Business Logic)
```
TranscriptionOrchestrator
├── Audio processing pipeline
│   ├── Chunk extraction with overlap
│   ├── Real-time transcription
│   └── Post-processing algorithms
├── State management
│   ├── Timestamp tracking
│   ├── Overlap resolution
│   └── Context accumulation
└── Algorithm selection
    ├── Simple overlap resolve
    └── Local agreement tracking
```

#### Supporting Business Logic
```
LocalAgreementTracker
├── Agreement counting
├── Edit distance calculation
└── Confidence thresholding

OverlapResolver
├── Word alignment algorithms
├── Timestamp adjustment
└── Conflict resolution
```

### 📱 **Application Layer (Use Cases)**

#### TranscriptionController
```
TranscriptionController (Use Case Coordinator)
├── Live transcription workflow
│   ├── Start/stop orchestration
│   ├── AudioDataProvider lifecycle
│   └── Error handling
├── File transcription
│   ├── Single file processing
│   └── Batch processing integration
└── Configuration updates
    ├── Dynamic parameter changes
    └── Runtime reconfiguration
```

#### BatchProcessor
```
BatchProcessor (Bulk Operations)
├── Folder scanning
├── Progress tracking
├── Error aggregation
└── Parallel processing support
```

### 🎨 **Presentation Layer (UI)**

#### MainWindow (Root UI Container)
```
MainWindow
├── Component orchestration
│   ├── Toolbar setup
│   ├── Status bar management
│   └── Window lifecycle
├── Feature integration
│   ├── Audio visualization
│   ├── Live text display
│   ├── Configuration dialogs
│   └── File operations
└── State coordination
    ├── ConfigManager integration
    ├── AppStateManager integration
    └── ThemeManager coordination
```

#### TranscriptionHandler (UI Business Logic)
```
TranscriptionHandler (UI Coordinator)
├── Model lifecycle management
│   ├── TranscriptionController initialization
│   ├── Start/stop operations
│   └── Resource cleanup
├── User interactions
│   ├── Recording toggle
│   ├── File selection
│   └── Batch processing
└── UI state synchronization
    ├── Status updates
    ├── Button state management
    └── Tray icon updates
```

#### Specialized UI Components
```
AudioVisualizer          # Real-time waveform display
LiveTextViewer           # Transcription output display
ConfigWindow             # Settings management
LoggingWindow            # Debug information
ThemeManager             # UI theming system
TrayIconManager          # System tray integration
```

## Current Architecture Issues & Dependencies

### 🔄 **Dependency Flow Analysis**

#### Clean Dependencies (Good)
```
MainWindow → TranscriptionHandler → TranscriptionController
                                      ↓
TranscriptionController → TranscriptionOrchestrator → TranscriptionEngine
TranscriptionController → AudioDataProvider → AudioCapture
```

#### Problematic Dependencies (Architectural Smells)

1. **UI Layer Coupling**
```
MainWindow → ConfigManager (Direct infrastructure access)
MainWindow → AppStateManager (Direct infrastructure access)
TranscriptionHandler → ConfigManager (UI layer accessing infra)
```

2. **Business Logic Leakage**
```
TranscriptionOrchestrator → AudioDataProvider (Should be injected)
TranscriptionOrchestrator → DocumentWriter (Direct I/O coupling)
```

3. **Cross-Layer Data Access**
```
UI Components → Business Configuration (TranscriptionConfig)
Presentation Layer → Infrastructure Models
```

### 🏛️ **Clean Architecture Refactoring Opportunities**

#### 1. **Dependency Inversion Principle Violations**

**Current:** Infrastructure depends on domain
```python
# Infrastructure reaching into domain
class ConfigManager:
    def get_config_dict(self) -> Dict[str, Any]:  # Returns raw dict
        # Domain logic mixed with I/O
```

**Proposed:** Domain defines interfaces
```python
# Domain defines abstraction
class ConfigRepository(ABC):
    @abstractmethod
    def get_transcription_config(self) -> TranscriptionConfig:
        pass

# Infrastructure implements interface
class JsonConfigRepository(ConfigRepository):
    def get_transcription_config(self) -> TranscriptionConfig:
        # I/O logic only
        pass
```

#### 2. **Presentation Layer Concerns**

**Current:** UI components know too much
```python
class MainWindow:
    def __init__(self, root):
        self.config_manager = ConfigManager()  # Infrastructure knowledge
        self.state_manager = AppStateManager() # Infrastructure knowledge
```

**Proposed:** Dependency injection with application services
```python
class MainWindow:
    def __init__(self, root, config_service, ui_state_service):
        self.config_service = config_service
        self.ui_state_service = ui_state_service
```

#### 3. **Business Logic Organization**

**Current:** Complex orchestrator with multiple responsibilities
```python
class TranscriptionOrchestrator:
    # Audio processing + State management + File I/O + Algorithm selection
```

**Proposed:** Decomposed business logic
```python
class TranscriptionOrchestrator:
    def __init__(self,
                 audio_processor: AudioProcessor,
                 transcription_service: TranscriptionService,
                 output_writer: TranscriptionWriter):
        # Single responsibility: coordinate the transcription workflow
```

### 📊 **Architectural Metrics**

#### Current Coupling Metrics
- **Afferent Coupling**: UI layer coupled to 4 infrastructure components
- **Efferent Coupling**: Business layer depends on 3 external systems
- **Abstractness**: Low (mostly concrete implementations)
- **Instability**: High (many dependencies)

#### Proposed Improvements
- **Interface Segregation**: Extract specific interfaces for each use case
- **Dependency Injection**: Use DI container for cross-cutting concerns
- **CQRS Pattern**: Separate read/write models for UI state
- **Repository Pattern**: Abstract data access behind domain interfaces

### 🎯 **Strategic Refactoring Roadmap**

#### Phase 1: Infrastructure Abstraction
1. Create repository interfaces in domain layer
2. Implement infrastructure adapters
3. Inject dependencies through application layer

#### Phase 2: Business Logic Decomposition
1. Extract transcription algorithms into strategy pattern
2. Separate state management from processing logic
3. Create domain services for complex operations

#### Phase 3: Presentation Layer Cleanup
1. Create view models for UI state management
2. Implement MVP/MVVM pattern
3. Remove infrastructure dependencies from UI

#### Phase 4: Cross-Cutting Concerns
1. Add logging abstraction
2. Implement error handling pipeline
3. Create configuration management system

This architecture analysis provides a foundation for systematic refactoring towards Clean Architecture principles while maintaining current functionality.
