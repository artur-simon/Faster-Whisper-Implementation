"""
Exemplo completo de uso do framework de testes de transcrição.

Este arquivo demonstra todos os recursos disponíveis.
"""

import logging
from pathlib import Path

from app.models import TranscriptionConfig
from tests.simulation.transcription_test_harness import (
    TranscriptionTestHarness,
    TranscriptionEvent
)
from tests.simulation.result_analyzer import (
    TranscriptionResultAnalyzer,
    TranscriptionComparator
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_1_basic_test():
    """Exemplo 1: Teste básico de transcrição"""
    
    print("\n" + "="*60)
    print("EXEMPLO 1: TESTE BÁSICO")
    print("="*60)
    
    config = TranscriptionConfig(
        model_size="base",
        device="cpu",
        compute_type="int8",
        sample_rate=16000,
        chunk_duration=5.0,
        overlap_duration=1.0,
        vad_filter=True,
        no_speech_threshold=0.6,
        language="auto",
        mic_id=0
    )
    
    harness = TranscriptionTestHarness(config)
    
    audio_file = "tests/fixtures/sample.wav"
    
    result = harness.run_test(audio_file, realtime_speed=10.0)
    
    print(f"Total de palavras: {len(result.all_words)}")
    print(f"Total de chunks: {result.total_chunks}")
    print(f"Texto final: {result.final_text[:100]}...")


def example_2_validation():
    """Exemplo 2: Validação de overlaps e gaps"""
    
    print("\n" + "="*60)
    print("EXEMPLO 2: VALIDAÇÃO DE QUALIDADE")
    print("="*60)
    
    config = TranscriptionConfig(
        model_size="base",
        device="cpu",
        compute_type="int8",
        sample_rate=16000,
        chunk_duration=5.0,
        overlap_duration=1.0,
        vad_filter=True,
        no_speech_threshold=0.6,
        language="auto",
        mic_id=0
    )
    
    harness = TranscriptionTestHarness(config)
    result = harness.run_test("tests/fixtures/sample.wav", realtime_speed=50.0)
    
    overlap_issues = result.find_overlap_issues()
    print(f"\nProblemas de overlap: {len(overlap_issues)}")
    
    if overlap_issues:
        print("Primeiros 3 overlaps:")
        for issue in overlap_issues[:3]:
            print(f"  '{issue['word1']}' sobrepõe '{issue['word2']}' "
                  f"por {issue['overlap_duration']:.3f}s")
    
    gaps = result.find_timing_gaps(threshold=1.5)
    print(f"\nGaps grandes (>1.5s): {len(gaps)}")
    
    if gaps:
        print("Primeiros 3 gaps:")
        for gap in gaps[:3]:
            print(f"  {gap['gap_duration']:.2f}s de silêncio após '{gap['after_word']}'")
    
    low_confidence = [w for w in result.all_words if w.probability < 0.5]
    print(f"\nPalavras com baixa confiança (<0.5): {len(low_confidence)}")


def example_3_event_tracking():
    """Exemplo 3: Rastreamento de eventos em tempo real"""
    
    print("\n" + "="*60)
    print("EXEMPLO 3: EVENT TRACKING")
    print("="*60)
    
    config = TranscriptionConfig(
        model_size="base",
        device="cpu",
        compute_type="int8",
        sample_rate=16000,
        chunk_duration=3.0,
        overlap_duration=0.5,
        vad_filter=True,
        no_speech_threshold=0.6,
        language="auto",
        mic_id=0
    )
    
    harness = TranscriptionTestHarness(config)
    
    event_log = []
    
    def event_callback(event: TranscriptionEvent):
        event_log.append(event)
        
        if event.event_type == "chunk_start":
            print(f"\n[Chunk: Processando...")
        
        elif event.event_type == "words_written":
            text = " ".join(w.text for w in event.words)
            print(f"[Chunk Escrito: {text}")
        
        elif event.event_type == "resolved_words":
            prev_text = event.metadata.get("previous_text", "")
            curr_text = event.metadata.get("current_text", "")
            resolved_text = event.metadata.get("resolved_text", "")
            
            if prev_text != resolved_text:
                print(f"[Chunk Overlap resolvido:")
                print(f"  Anterior: {prev_text}")
                print(f"  Atual: {curr_text}")
                print(f"  Resolvido: {resolved_text}")
    
    harness.set_event_callback(event_callback)
    
    result = harness.run_test("tests/fixtures/sample.wav", realtime_speed=20.0)
    
    print(f"\nTotal de eventos capturados: {len(event_log)}")
    
    event_types = {}
    for event in event_log:
        event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
    
    print("\nDistribuição de eventos:")
    for event_type, count in event_types.items():
        print(f"  {event_type}: {count}")


def example_4_full_analysis():
    """Exemplo 4: Análise completa com relatório"""
    
    print("\n" + "="*60)
    print("EXEMPLO 4: ANÁLISE COMPLETA")
    print("="*60)
    
    config = TranscriptionConfig(
        model_size="base",
        device="cpu",
        compute_type="int8",
        sample_rate=16000,
        chunk_duration=5.0,
        overlap_duration=1.0,
        vad_filter=True,
        no_speech_threshold=0.6,
        language="auto",
        mic_id=0
    )
    
    harness = TranscriptionTestHarness(config)
    result = harness.run_test("tests/fixtures/sample.wav", realtime_speed=50.0)
    
    TranscriptionResultAnalyzer.print_summary(result)
    
    output_path = "test_report.json"
    TranscriptionResultAnalyzer.save_report_to_file(result, output_path)
    print(f"Relatório completo salvo em: {output_path}")


def example_5_compare_configs():
    """Exemplo 5: Comparar diferentes configurações"""
    
    print("\n" + "="*60)
    print("EXEMPLO 5: COMPARAÇÃO DE CONFIGURAÇÕES")
    print("="*60)
    
    base_config = TranscriptionConfig(
        model_size="base",
        device="cpu",
        compute_type="int8",
        sample_rate=16000,
        chunk_duration=5.0,
        overlap_duration=1.0,
        vad_filter=True,
        no_speech_threshold=0.6,
        language="auto",
        mic_id=0
    )
    
    configs_to_test = [
        {"name": "Chunk curto", "chunk_duration": 3.0, "overlap_duration": 0.5},
        {"name": "Chunk médio", "chunk_duration": 5.0, "overlap_duration": 1.0},
        {"name": "Chunk longo", "chunk_duration": 10.0, "overlap_duration": 2.0},
    ]
    
    results = []
    
    for cfg in configs_to_test:
        config = TranscriptionConfig(
            model_size=base_config.model_size,
            device=base_config.device,
            compute_type=base_config.compute_type,
            sample_rate=base_config.sample_rate,
            chunk_duration=cfg["chunk_duration"],
            overlap_duration=cfg["overlap_duration"],
            vad_filter=base_config.vad_filter,
            no_speech_threshold=base_config.no_speech_threshold,
            language=base_config.language,
            mic_id=base_config.mic_id
        )
        
        harness = TranscriptionTestHarness(config)
        result = harness.run_test("tests/fixtures/sample.wav", realtime_speed=50.0)
        
        overlap_count = len(result.find_overlap_issues())
        gap_count = len(result.find_timing_gaps(threshold=1.0))
        
        results.append({
            "config": cfg,
            "result": result,
            "overlaps": overlap_count,
            "gaps": gap_count
        })
        
        print(f"\n{cfg['name']}:")
        print(f"  Chunk: {cfg['chunk_duration']}s, Overlap: {cfg['overlap_duration']}s")
        print(f"  Palavras: {len(result.all_words)}")
        print(f"  Chunks processados: {result.total_chunks}")
        print(f"  Overlaps: {overlap_count}")
        print(f"  Gaps: {gap_count}")
        print(f"  Texto: {result.final_text[:80]}...")
    
    if len(results) >= 2:
        print("\n" + "-"*60)
        print("COMPARAÇÃO ENTRE CONFIGS:")
        print("-"*60)
        
        comparison = TranscriptionComparator.compare_results(
            results[0]["result"],
            results[1]["result"]
        )
        
        print(f"Diferença de palavras: {comparison['word_count_diff']}")
        print(f"Diferença de chunks: {comparison['chunk_count_diff']}")
        print(f"Similaridade de texto: {comparison['text_similarity']:.2%}")


def example_6_specific_time_range():
    """Exemplo 6: Analisar intervalo de tempo específico"""
    
    print("\n" + "="*60)
    print("EXEMPLO 6: ANÁLISE DE INTERVALO DE TEMPO")
    print("="*60)
    
    config = TranscriptionConfig(
        model_size="base",
        device="cpu",
        compute_type="int8",
        sample_rate=16000,
        chunk_duration=5.0,
        overlap_duration=1.0,
        vad_filter=True,
        no_speech_threshold=0.6,
        language="auto",
        mic_id=0
    )
    
    harness = TranscriptionTestHarness(config)
    result = harness.run_test("tests/fixtures/sample.wav", realtime_speed=50.0)
    
    if result.all_words:
        total_duration = result.all_words[-1].end
        
        time_ranges = [
            (0, 5),
            (5, 10),
            (10, 15)
        ]
        
        for start, end in time_ranges:
            if end <= total_duration:
                text = result.get_text_by_timerange(start, end)
                print(f"\n[{start}s - {end}s]: {text}")


def main():
    """Executa todos os exemplos"""
    
    audio_file = Path("tests/fixtures/sample.wav")
    if not audio_file.exists():
        print(f"AVISO: Arquivo de áudio não encontrado: {audio_file}")
        print("Crie um arquivo de teste ou ajuste o caminho nos exemplos.")
        return
    
    examples = [
        example_1_basic_test,
        example_2_validation,
        example_3_event_tracking,
        example_4_full_analysis,
        example_5_compare_configs,
        example_6_specific_time_range
    ]
    
    for i, example_func in enumerate(examples, 1):
        try:
            example_func()
        except Exception as e:
            print(f"\nErro no exemplo {i}: {e}")
            logger.exception(f"Erro detalhado no exemplo {i}")
    
    print("\n" + "="*60)
    print("TODOS OS EXEMPLOS CONCLUÍDOS")
    print("="*60)


if __name__ == "__main__":
    main()

