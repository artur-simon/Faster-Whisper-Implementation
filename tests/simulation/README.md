# Transcription Simulation Testing Framework

Sistema de testes que simula transcrição em tempo real usando arquivos de áudio, permitindo validação confiável e reprodutível de cenários problemáticos.

## Estrutura

- `audio_file_simulator.py` - Simula captura de áudio em tempo real a partir de arquivos
- `transcription_test_harness.py` - Framework principal de testes com captura de eventos
- `result_analyzer.py` - Análise detalhada de resultados (overlaps, timing, qualidade)
- `test_real_audio_scenarios.py` - Casos de teste automatizados
- `run_manual_test.py` - Script standalone para testes manuais

## Uso Rápido

### Teste Manual

```bash
python tests/simulation/run_manual_test.py tests/fixtures/sample.wav report.json
```


## Validações Disponíveis

### Overlaps de Timestamps

```python
overlap_issues = result.find_overlap_issues()
```

Detecta palavras com timestamps sobrepostos (palavra1.end > palavra2.start).

### Gaps de Tempo

```python
gaps = result.find_timing_gaps(threshold=2.0)
```

Identifica pausas excessivas entre palavras.

## Event Tracking

```python
def event_callback(event):
    if event.event_type == "words_written":
        print(f"Chunk : {' '.join(w.text for w in event.words)}")

harness.set_event_callback(event_callback)
result = harness.run_test("audio.wav")
```

Tipos de eventos:
- `chunk_start` - Início do processamento de chunk
- `raw_transcription` - Transcrição bruta do Whisper
- `resolved_words` - Palavras após resolução de overlap
- `words_written` - Palavras escritas no resultado final
- `error` - Erro durante processamento


## Estrutura do Resultado

```python
@dataclass
class TranscriptionTestResult:
    all_words: List[Word]
    events: List[TranscriptionEvent]
    final_text: str
    total_chunks: int
    total_duration: float
    
    def get_text_by_timerange(self, start: float, end: float) -> str
    def find_overlap_issues(self) -> List[dict]
    def find_timing_gaps(self, threshold: float = 1.0) -> List[dict]
```

## Comparação de Resultados

```python
from tests.simulation.result_analyzer import TranscriptionComparator

result1 = harness1.run_test("audio.wav")
result2 = harness2.run_test("audio.wav")

comparison = TranscriptionComparator.compare_results(result1, result2)

print(f"Word count diff: {comparison['word_count_diff']}")
print(f"Text similarity: {comparison['text_similarity']:.2f}")
```