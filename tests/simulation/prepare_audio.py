"""
Script auxiliar para preparar arquivos de áudio para testes.

Converte diferentes formatos de áudio para WAV 16kHz mono,
que é o formato ideal para os testes.
"""

import wave
import sys
import subprocess
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_ffmpeg_available():
    """Verifica se ffmpeg está disponível no sistema"""
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def get_wav_info(wav_path: str):
    """Obtém informações de um arquivo WAV"""
    try:
        with wave.open(wav_path, 'rb') as wf:
            return {
                'sample_rate': wf.getframerate(),
                'channels': wf.getnchannels(),
                'sample_width': wf.getsampwidth(),
                'frames': wf.getnframes(),
                'duration': wf.getnframes() / wf.getframerate()
            }
    except Exception as e:
        logger.error(f"Erro ao ler arquivo WAV: {e}")
        return None


def convert_to_test_format(input_path: str, output_path: str = None):
    """
    Converte arquivo de áudio para formato de teste (WAV 16kHz mono).
    
    Args:
        input_path: Caminho do arquivo de entrada
        output_path: Caminho do arquivo de saída (opcional)
    
    Returns:
        Caminho do arquivo convertido ou None se houver erro
    """
    input_file = Path(input_path)
    
    if not input_file.exists():
        logger.error(f"Arquivo não encontrado: {input_path}")
        return None
    
    if output_path is None:
        output_path = input_file.parent / f"{input_file.stem}_16k.wav"
    
    output_file = Path(output_path)
    
    if not check_ffmpeg_available():
        logger.error("ffmpeg não encontrado. Instale com: pip install ffmpeg-python ou baixe de ffmpeg.org")
        return None
    
    logger.info(f"Convertendo {input_file.name} para formato de teste...")
    
    try:
        cmd = [
            'ffmpeg',
            '-i', str(input_file),
            '-ar', '16000',
            '-ac', '1',
            '-y',
            str(output_file)
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"Erro na conversão: {result.stderr}")
            return None
        
        logger.info(f"Arquivo convertido: {output_file}")
        
        info = get_wav_info(str(output_file))
        if info:
            logger.info(f"  Sample rate: {info['sample_rate']} Hz")
            logger.info(f"  Canais: {info['channels']}")
            logger.info(f"  Duração: {info['duration']:.2f}s")
        
        return str(output_file)
        
    except Exception as e:
        logger.error(f"Erro durante conversão: {e}")
        return None


def validate_audio_file(audio_path: str):
    """
    Valida se arquivo de áudio está no formato correto para testes.
    
    Args:
        audio_path: Caminho do arquivo
    
    Returns:
        tuple (is_valid, issues)
    """
    audio_file = Path(audio_path)
    
    if not audio_file.exists():
        return False, ["Arquivo não encontrado"]
    
    if audio_file.suffix.lower() != '.wav':
        return False, ["Arquivo não é WAV (use convert_to_test_format)"]
    
    info = get_wav_info(audio_path)
    
    if info is None:
        return False, ["Não foi possível ler arquivo WAV"]
    
    issues = []
    
    if info['sample_rate'] != 16000:
        issues.append(f"Sample rate {info['sample_rate']} Hz (recomendado: 16000 Hz)")
    
    if info['channels'] != 1:
        issues.append(f"{info['channels']} canais (recomendado: 1 - mono)")
    
    if info['duration'] < 1.0:
        issues.append(f"Duração muito curta: {info['duration']:.2f}s")
    
    if info['duration'] > 600:
        logger.warning(f"Áudio longo ({info['duration']:.0f}s) - teste pode demorar")
    
    is_valid = len(issues) == 0
    
    if is_valid:
        logger.info(f"Arquivo válido: {audio_file.name}")
        logger.info(f"  Duração: {info['duration']:.2f}s")
    else:
        logger.warning(f"Arquivo tem problemas:")
        for issue in issues:
            logger.warning(f"  - {issue}")
    
    return is_valid, issues


def prepare_test_audio(input_path: str, output_dir: str = None):
    """
    Prepara arquivo de áudio para teste (valida ou converte se necessário).
    
    Args:
        input_path: Arquivo de entrada
        output_dir: Diretório de saída (padrão: tests/fixtures)
    
    Returns:
        Caminho do arquivo pronto para teste
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "fixtures"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    input_file = Path(input_path)
    
    is_wav = input_file.suffix.lower() == '.wav'
    
    if is_wav:
        is_valid, issues = validate_audio_file(input_path)
        
        if is_valid:
            logger.info(f"Arquivo já está no formato correto: {input_path}")
            return input_path
    
    output_path = output_dir / f"{input_file.stem}_test.wav"
    
    converted = convert_to_test_format(input_path, str(output_path))
    
    if converted:
        logger.info(f"Arquivo pronto para teste: {converted}")
        return converted
    else:
        logger.error("Falha ao preparar arquivo")
        return None


def main():
    if len(sys.argv) < 2:
        print("Uso: python prepare_audio.py <audio_file> [output_file]")
        print("\nExemplos:")
        print("  python prepare_audio.py audio.mp3")
        print("  python prepare_audio.py audio.mp3 tests/fixtures/test.wav")
        print("  python prepare_audio.py audio.wav  # Valida formato")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    input_file = Path(input_path)
    
    if not input_file.exists():
        print(f"Erro: Arquivo não encontrado: {input_path}")
        sys.exit(1)
    
    if input_file.suffix.lower() == '.wav':
        print(f"\nValidando arquivo WAV: {input_path}")
        is_valid, issues = validate_audio_file(input_path)
        
        if is_valid:
            print("\n✓ Arquivo está pronto para testes!")
        else:
            print("\n✗ Arquivo precisa ser convertido:")
            for issue in issues:
                print(f"  - {issue}")
            
            convert = input("\nConverter agora? (s/n): ").lower() == 's'
            if convert:
                result = prepare_test_audio(input_path, Path(output_path).parent if output_path else None)
                if result:
                    print(f"\n✓ Arquivo convertido: {result}")
                else:
                    print("\n✗ Erro na conversão")
                    sys.exit(1)
    else:
        print(f"\nConvertendo {input_file.suffix} para formato de teste...")
        result = convert_to_test_format(input_path, output_path)
        
        if result:
            print(f"\n✓ Sucesso! Arquivo pronto: {result}")
        else:
            print("\n✗ Erro na conversão")
            sys.exit(1)


if __name__ == "__main__":
    main()

