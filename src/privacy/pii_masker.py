"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    PII Masker: Proteção de Dados Pessoais                    ║
║                                                                              ║
║  Este arquivo remove informações pessoais (nome, CPF, telefone, etc.) de    ║
║  textos antes de enviar para a IA processar. É ESSENCIAL para privacidade!  ║
╚══════════════════════════════════════════════════════════════════════════════╝

🎯 O QUE ESTE ARQUIVO FAZ?
---------------------------
Quando um paciente faz uma consulta, ele pode dizer coisas como:
"Olá, sou Maria Silva, CPF 123.456.789-00, e estou com dor de cabeça"

Não podemos enviar esses dados pessoais para a IA! Por isso:
1. Identificamos dados pessoais (PII = Personally Identifiable Information)
2. Substituímos por tokens: [PERSON_1], [CPF_1], etc.
3. Enviamos o texto "limpo" para a IA

Resultado:
"Olá, sou [PERSON_1], CPF [CPF_1], e estou com dor de cabeça"

📚 CONCEITOS IMPORTANTES:
-------------------------
- PII: Personally Identifiable Information (dados que identificam uma pessoa)
- LGPD: Lei Geral de Proteção de Dados (lei brasileira de privacidade)
- HIPAA: Lei americana de proteção de dados de saúde
- NER: Named Entity Recognition (IA que identifica nomes, lugares, etc.)
- REGEX: Expressões regulares (padrões para encontrar texto)

🔗 DEPENDÊNCIAS:
----------------
- re: Biblioteca de expressões regulares (padrão Python)
- hashlib: Para criar hashes (códigos únicos)
- spacy: Biblioteca de NLP para identificar entidades (opcional)
- pydantic: Validação de dados
"""

# ============================================================================
# IMPORTS
# ============================================================================

from __future__ import annotations

# re: Regular Expressions (expressões regulares)
# Permite buscar padrões em texto, como "3 dígitos + ponto + 3 dígitos"
import re

# hashlib: Para criar "hashes" (códigos únicos e irreversíveis)
# Útil quando queremos esconder um dado mas ainda identificá-lo unicamente
import hashlib

# typing: Define tipos das variáveis
from typing import NamedTuple

# dataclass: Cria classes simples para armazenar dados
from dataclasses import dataclass, field

# functools: Ferramentas para funções (usamos lru_cache para otimização)
from functools import lru_cache

# Pydantic: Valida dados automaticamente
from pydantic import BaseModel


# ============================================================================
# DEFINIÇÃO DE ENTIDADES
# ============================================================================

class PIIEntity(BaseModel):
    """
    🔐 Representa UMA entidade PII encontrada no texto.
    
    Quando encontramos um dado pessoal, criamos um objeto deste tipo
    para registrar o que foi encontrado e como foi mascarado.
    
    Exemplo:
        >>> entity = PIIEntity(
        ...     entity_type="CPF",
        ...     original_value="123.456.789-00",
        ...     masked_value="[CPF_1]",
        ...     start_pos=33,
        ...     end_pos=47,
        ...     confidence=0.95
        ... )
    """
    entity_type: str        # Tipo: "CPF", "PHONE", "EMAIL", "PERSON", etc.
    original_value: str     # Valor original encontrado
    masked_value: str       # O token que substituiu (ex: [CPF_1])
    start_pos: int          # Posição inicial no texto (caractere)
    end_pos: int            # Posição final no texto
    confidence: float       # Confiança da detecção (0.0 a 1.0)


@dataclass
class MaskingResult:
    """
    📋 Resultado completo do processo de mascaramento.
    
    Contém o texto original, o texto mascarado, e todas as entidades
    que foram encontradas e substituídas.
    
    Exemplo:
        >>> result = masker.mask("Maria Silva, CPF 123.456.789-00")
        >>> print(result.masked_text)
        "[PERSON_1], CPF [CPF_1]"
        >>> print(result.entity_count)
        2
    """
    original_text: str                           # Texto original
    masked_text: str                             # Texto com PII removido
    entities_found: list[PIIEntity] = field(default_factory=list)  # Lista de PIIs
    pii_detected: bool = False                   # True se encontrou algum PII
    
    @property
    def entity_count(self) -> int:
        """Conta quantas entidades PII foram encontradas."""
        return len(self.entities_found)
    
    def get_entity_map(self) -> dict[str, str]:
        """
        Retorna mapeamento token → valor original.
        
        Útil para auditoria (se precisar saber o que foi mascarado).
        ⚠️ CUIDADO: Este mapeamento contém dados sensíveis!
        
        Returns:
            {"[CPF_1]": "123.456.789-00", "[PERSON_1]": "Maria Silva"}
        """
        return {e.masked_value: e.original_value for e in self.entities_found}


# ============================================================================
# PADRÕES REGEX
# ============================================================================
# Expressões regulares (regex) são "receitas" para encontrar padrões em texto.
# Por exemplo: \d{3}\.\d{3}\.\d{3}-\d{2} encontra CPF no formato 000.000.000-00
#
# Símbolos importantes:
#   \d     = qualquer dígito (0-9)
#   \d{3}  = exatamente 3 dígitos
#   \.     = ponto literal (sem \, o ponto significa "qualquer caractere")
#   [.-]?  = ponto ou hífen, opcional (? = 0 ou 1 vez)
#   \b     = limite de palavra (não pode ter letra/número antes ou depois)
#   |      = OU (uma coisa ou outra)
#   (?:...)= grupo que não captura (agrupa mas não cria grupo de captura)
# ============================================================================

class PIIPatterns:
    """
    📝 Padrões regex para detectar PII em texto brasileiro.
    
    Cada padrão é uma "receita" que o computador usa para encontrar
    tipos específicos de dados pessoais no texto.
    """
    
    # =========================================================================
    # CPF: 000.000.000-00 ou 00000000000
    # =========================================================================
    # Explicação do regex:
    #   \b                  = limite de palavra
    #   \d{3}               = 3 dígitos
    #   \.?                 = ponto opcional
    #   \d{3}               = mais 3 dígitos
    #   \.?                 = ponto opcional
    #   \d{3}               = mais 3 dígitos
    #   [-.]?               = hífen ou ponto opcional
    #   \d{2}               = 2 dígitos finais
    #   \b                  = limite de palavra
    #
    # Exemplos que encontra: "123.456.789-00", "12345678900", "123.456.789.00"
    CPF = re.compile(
        r'\b\d{3}\.?\d{3}\.?\d{3}[-.]?\d{2}\b'
    )
    
    # =========================================================================
    # RG: Varia por estado, padrão genérico
    # =========================================================================
    # Formato aproximado: 1-2 dígitos + 3 dígitos + 3 dígitos + 1 dígito ou X
    # Exemplos: "12.345.678-9", "1.234.567-X"
    RG = re.compile(
        r'\b\d{1,2}\.?\d{3}\.?\d{3}[-.]?[\dXx]\b'
    )
    
    # =========================================================================
    # TELEFONE: (11) 99999-9999 ou +55 11 99999-9999
    # =========================================================================
    # Partes:
    #   (?:\+55\s?)?        = opcional: +55 com espaço opcional
    #   (?:\(?\d{2}\)?...)?  = opcional: DDD com ou sem parênteses
    #   \d{4,5}             = 4 ou 5 dígitos (celular tem 9 dígitos)
    #   [-.]?               = hífen ou ponto opcional
    #   \d{4}               = 4 dígitos finais
    PHONE = re.compile(
        r'(?:\+55\s?)?(?:\(?\d{2}\)?[\s.-]?)?\d{4,5}[-.]?\d{4}\b'
    )
    
    # =========================================================================
    # EMAIL: usuario@dominio.com
    # =========================================================================
    # Partes:
    #   [A-Za-z0-9._%+-]+   = nome do usuário (letras, números, alguns símbolos)
    #   @                    = arroba
    #   [A-Za-z0-9.-]+      = domínio (letras, números, ponto, hífen)
    #   \.                  = ponto
    #   [A-Z|a-z]{2,}       = extensão (com, br, etc.) - mínimo 2 letras
    EMAIL = re.compile(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    )
    
    # =========================================================================
    # CEP: 00000-000
    # =========================================================================
    CEP = re.compile(
        r'\b\d{5}[-.]?\d{3}\b'
    )
    
    # =========================================================================
    # DATA DE NASCIMENTO (em contexto médico)
    # =========================================================================
    # Procura por frases como "nascido em 15/03/1985"
    # O (\d{1,2}[/-]\d{1,2}[/-]\d{2,4}) é um GRUPO DE CAPTURA
    # Isso significa que só a DATA será extraída, não o texto "nascido em"
    BIRTH_DATE = re.compile(
        r'\b(?:nascido|nascimento|nasc\.?|data de nascimento)[:\s]*'
        r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',
        re.IGNORECASE  # Ignora maiúsculas/minúsculas
    )
    
    # =========================================================================
    # ENDEREÇO: Rua das Flores, 123
    # =========================================================================
    # Procura padrões comuns de endereço brasileiro
    ADDRESS = re.compile(
        r'\b(?:rua|av\.?|avenida|alameda|travessa|praça)\s+[A-Za-zÀ-ÿ\s]+,?\s*'
        r'(?:n[°ºo]?\.?\s*)?\d+',
        re.IGNORECASE
    )
    
    # =========================================================================
    # NOME PRÓPRIO (heurística simples)
    # =========================================================================
    # Procura sequências de palavras que começam com maiúscula
    # Ex: "Sr. João da Silva", "Maria Santos"
    # ⚠️ Cuidado: pode ter falsos positivos! Por isso usamos NER também.
    PROPER_NAME = re.compile(
        r'\b(?:(?:sr\.?|sra\.?|dr\.?|dra\.?)\s+)?'
        r'(?:[A-ZÀ-Ú][a-zà-ú]+\s+){1,4}[A-ZÀ-Ú][a-zà-ú]+\b'
    )


# ============================================================================
# CLASSE PRINCIPAL DO MASCARADOR
# ============================================================================

class PIIMasker:
    """
    🎭 Mascarador de PII com múltiplas estratégias.
    
    Esta é a classe principal que você usa para remover dados pessoais.
    
    ESTRATÉGIAS DISPONÍVEIS:
    
    1. TOKEN (padrão): Substitui por tokens genéricos
       "Maria Silva" → "[PERSON_1]"
       "123.456.789-00" → "[CPF_1]"
       ✅ Bom para: Anonimização completa
    
    2. HASH: Substitui por código único (determinístico)
       "Maria Silva" → "[HASH_a1b2c3d4e5f6]"
       ✅ Bom para: Quando precisa identificar a mesma pessoa depois
       
    3. REDACT: Remove completamente
       "Maria Silva" → "[REDACTED]"
       ✅ Bom para: Máxima segurança
    
    EXEMPLO DE USO:
    
        >>> # Criar o mascarador
        >>> masker = PIIMasker(strategy="TOKEN")
        >>> 
        >>> # Mascarar um texto
        >>> result = masker.mask("Meu CPF é 123.456.789-00")
        >>> print(result.masked_text)
        "Meu CPF é [CPF_1]"
        >>> 
        >>> # Ver o que foi encontrado
        >>> for entity in result.entities_found:
        ...     print(f"{entity.entity_type}: {entity.original_value}")
        CPF: 123.456.789-00
    """
    
    def __init__(
        self,
        strategy: str = "TOKEN",
        salt: str | None = None,
        enable_ner: bool = True,
    ):
        """
        Inicializa o mascarador.
        
        Args:
            strategy: "TOKEN", "HASH" ou "REDACT"
            salt: "Sal" para o hash (segurança extra, mude em produção!)
            enable_ner: Usar IA para detectar nomes (precisa do spaCy)
        
        Exemplo:
            >>> masker = PIIMasker(strategy="TOKEN", enable_ner=True)
        """
        # Converte para maiúsculo para padronizar
        self.strategy = strategy.upper()
        
        # Salt para hashing (adiciona aleatoriedade)
        # ⚠️ Em produção, use um salt secreto e único!
        self.salt = salt or "neurotriage-default-salt"
        
        # Se True, usa spaCy para detectar nomes (mais preciso)
        self.enable_ner = enable_ner
        
        # Contador para gerar tokens únicos: [CPF_1], [CPF_2], etc.
        self._token_counters: dict[str, int] = {}
        
        # Cache do modelo NER (carrega sob demanda)
        self._ner_model = None
        
        # Mapeamento tipo → padrão regex
        self.patterns = {
            "CPF": PIIPatterns.CPF,
            "RG": PIIPatterns.RG,
            "PHONE": PIIPatterns.PHONE,
            "EMAIL": PIIPatterns.EMAIL,
            "CEP": PIIPatterns.CEP,
            "BIRTH_DATE": PIIPatterns.BIRTH_DATE,
            "ADDRESS": PIIPatterns.ADDRESS,
        }
    
    @lru_cache(maxsize=1)
    def _get_ner_model(self):
        """
        🧠 Carrega o modelo NER (Named Entity Recognition).
        
        NER é uma IA que identifica "entidades" em texto:
        "Maria Silva foi a São Paulo" 
        → PESSOA: Maria Silva
        → LOCAL: São Paulo
        
        Usamos o modelo pt_core_news_lg do spaCy (português).
        
        @lru_cache: Guarda o modelo em memória depois de carregar
        (não precisa carregar de novo a cada chamada)
        
        Returns:
            Modelo spaCy carregado, ou None se não disponível
        """
        if not self.enable_ner:
            return None
        
        try:
            import spacy
            return spacy.load("pt_core_news_lg")
        except OSError:
            # Modelo não instalado, vamos baixar
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "pt_core_news_lg"])
            import spacy
            return spacy.load("pt_core_news_lg")
    
    def _generate_token(self, entity_type: str) -> str:
        """
        🏷️ Gera um token único para o tipo de entidade.
        
        Cada tipo tem seu próprio contador:
        - Primeiro CPF: [CPF_1]
        - Segundo CPF: [CPF_2]
        - Primeiro PHONE: [PHONE_1]
        
        Args:
            entity_type: "CPF", "PHONE", "PERSON", etc.
        
        Returns:
            Token no formato [TIPO_NUMERO]
        """
        # Pega o contador atual (ou 0 se não existe) e incrementa
        counter = self._token_counters.get(entity_type, 0) + 1
        self._token_counters[entity_type] = counter
        return f"[{entity_type}_{counter}]"
    
    def _generate_hash(self, value: str) -> str:
        """
        🔒 Gera um hash determinístico do valor.
        
        Hash é uma função que transforma qualquer texto em um código fixo.
        Características:
        - Mesmo texto → sempre mesmo hash
        - Impossível reverter (não dá pra descobrir o original)
        - Mudança mínima no texto → hash completamente diferente
        
        Usamos SHA-256 que é seguro e rápido.
        
        Args:
            value: O valor a ser hasheado (ex: "Maria Silva")
        
        Returns:
            Token no formato [HASH_abc123def456]
        
        Exemplo:
            "Maria Silva" → [HASH_a1b2c3d4e5f6]
            "Maria Silva" → [HASH_a1b2c3d4e5f6]  # Mesmo resultado!
            "maria silva" → [HASH_x9y8z7w6v5u4]  # Diferente!
        """
        # Adiciona o salt para segurança extra
        salted = f"{self.salt}:{value}"
        
        # Calcula o hash SHA-256 e pega só os primeiros 12 caracteres
        hash_bytes = hashlib.sha256(salted.encode()).hexdigest()[:12]
        
        return f"[HASH_{hash_bytes}]"
    
    def _mask_value(self, value: str, entity_type: str) -> str:
        """
        🎭 Aplica a estratégia de mascaramento ao valor.
        
        Baseado na estratégia escolhida na inicialização:
        - TOKEN: [CPF_1], [PERSON_2], etc.
        - HASH: [HASH_abc123], etc.
        - REDACT: [REDACTED]
        
        Args:
            value: O valor original
            entity_type: Tipo da entidade
        
        Returns:
            O valor mascarado
        """
        # match/case é como um switch em outras linguagens
        # Compara self.strategy com cada opção
        match self.strategy:
            case "TOKEN":
                return self._generate_token(entity_type)
            case "HASH":
                return self._generate_hash(value)
            case "REDACT":
                return "[REDACTED]"
            case _:
                raise ValueError(f"Estratégia desconhecida: {self.strategy}")
    
    def _detect_names_ner(self, text: str) -> list[tuple[int, int, str]]:
        """
        🔍 Detecta nomes próprios usando IA (NER).
        
        O spaCy processa o texto e identifica entidades como:
        - PER/PERSON: Pessoas ("Maria Silva", "Dr. João")
        - LOC: Locais ("São Paulo", "Brasil")
        - ORG: Organizações ("Google", "Governo Federal")
        
        Só nos interessam pessoas (PER/PERSON).
        
        Args:
            text: Texto a analisar
        
        Returns:
            Lista de tuplas: (posição_inicial, posição_final, texto_nome)
        """
        if not self.enable_ner:
            return []
        
        nlp = self._get_ner_model()
        if nlp is None:
            return []
        
        # Processa o texto
        doc = nlp(text)
        
        names = []
        for ent in doc.ents:  # ent = entity (entidade)
            # Só queremos pessoas
            if ent.label_ in ("PER", "PERSON"):
                names.append((ent.start_char, ent.end_char, ent.text))
        
        return names
    
    def mask(self, text: str) -> MaskingResult:
        """
        🎭 FUNÇÃO PRINCIPAL: Mascara todas as PII no texto.
        
        Este é o método que você vai usar na maioria das vezes.
        Ele encontra TODOS os dados pessoais e substitui por tokens.
        
        Args:
            text: Texto original com possíveis dados pessoais
        
        Returns:
            MaskingResult com:
            - masked_text: Texto "limpo" (sem dados pessoais)
            - entities_found: Lista do que foi encontrado
            - pii_detected: True se encontrou algo
        
        COMO FUNCIONA (passo a passo):
        1. Reseta contadores de tokens
        2. Busca padrões com regex (CPF, telefone, email...)
        3. Busca nomes com NER (IA)
        4. Ordena entidades por posição
        5. Substitui de trás para frente (para não bagunçar posições)
        6. Retorna resultado
        
        Exemplo:
            >>> masker = PIIMasker()
            >>> result = masker.mask("Sou Maria Silva, CPF 123.456.789-00")
            >>> print(result.masked_text)
            "Sou [PERSON_1], CPF [CPF_1]"
        """
        # PASSO 1: Reseta contadores (cada chamada começa do zero)
        self._token_counters = {}
        
        # Lista de entidades encontradas
        entities: list[PIIEntity] = []
        
        # Começamos com o texto original (vamos modificar)
        masked_text = text
        
        # PASSO 2: Busca entidades com REGEX
        # Para cada tipo (CPF, PHONE, etc.) e seu padrão regex...
        for entity_type, pattern in self.patterns.items():
            # finditer retorna todos os matches encontrados
            for match in pattern.finditer(text):
                # Alguns padrões têm grupos de captura (ex: data dentro de "nascido em")
                if match.groups():
                    # Se tem grupo, pega o primeiro grupo
                    original = match.group(1)
                    start = match.start(1)
                    end = match.end(1)
                else:
                    # Se não tem grupo, pega o match inteiro
                    original = match.group()
                    start = match.start()
                    end = match.end()
                
                # Gera o token de substituição
                masked_value = self._mask_value(original, entity_type)
                
                # Adiciona à lista de entidades
                entities.append(PIIEntity(
                    entity_type=entity_type,
                    original_value=original,
                    masked_value=masked_value,
                    start_pos=start,
                    end_pos=end,
                    confidence=0.95,  # Regex tem alta precisão
                ))
        
        # PASSO 3: Busca NOMES com NER (IA)
        for start, end, name in self._detect_names_ner(text):
            # Evita duplicatas (se regex já pegou este texto)
            if any(e.start_pos == start for e in entities):
                continue
            
            masked_value = self._mask_value(name, "PERSON")
            
            entities.append(PIIEntity(
                entity_type="PERSON",
                original_value=name,
                masked_value=masked_value,
                start_pos=start,
                end_pos=end,
                confidence=0.85,  # NER tem confiança variável
            ))
        
        # PASSO 4: Ordena entidades por posição (DE TRÁS PARA FRENTE)
        # Por que de trás para frente?
        # Se substituímos algo no início, as posições do resto mudam!
        # Exemplo: "AB CD" → substituir "AB" por "[TOKEN]"
        # "CD" estava nas posições 3-4, mas agora está em 8-9!
        # Substituindo de trás para frente, as posições anteriores não mudam.
        entities.sort(key=lambda e: e.start_pos, reverse=True)
        
        # PASSO 5: Aplica substituições
        for entity in entities:
            # Fatia o texto: antes + token + depois
            masked_text = (
                masked_text[:entity.start_pos] +       # Texto antes
                entity.masked_value +                   # Token
                masked_text[entity.end_pos:]            # Texto depois
            )
        
        # PASSO 6: Reordena para output (ordem de aparição no texto)
        entities.sort(key=lambda e: e.start_pos)
        
        return MaskingResult(
            original_text=text,
            masked_text=masked_text,
            entities_found=entities,
            pii_detected=len(entities) > 0,
        )
    
    def validate_no_pii(self, text: str) -> bool:
        """
        ✅ Valida que um texto NÃO contém PII.
        
        Útil para verificar outputs da IA antes de salvar/mostrar.
        
        Args:
            text: Texto a verificar
        
        Returns:
            True se está "limpo" (sem PII)
            False se encontrou algum PII
        
        Exemplo:
            >>> masker.validate_no_pii("Olá, como vai?")
            True
            >>> masker.validate_no_pii("Meu CPF é 123.456.789-00")
            False
        """
        result = self.mask(text)
        return not result.pii_detected


# ============================================================================
# FUNÇÕES DE FÁBRICA
# ============================================================================
# "Fábricas" são funções que criam objetos pré-configurados.
# Facilita o uso em diferentes contextos (HIPAA vs LGPD).
# ============================================================================

def create_hipaa_masker() -> PIIMasker:
    """
    🏥 Cria mascarador para conformidade HIPAA (lei americana de saúde).
    
    Usa estratégia HASH para permitir "linking" em auditorias.
    (Se a mesma pessoa aparece em dois textos, terá o mesmo hash)
    
    Returns:
        PIIMasker configurado para HIPAA
    """
    return PIIMasker(
        strategy="HASH",      # Hash permite identificar mesma pessoa
        enable_ner=True,      # Usa IA para nomes
    )


def create_lgpd_masker() -> PIIMasker:
    """
    🇧🇷 Cria mascarador para conformidade LGPD (lei brasileira de dados).
    
    Usa estratégia TOKEN para anonimização completa.
    (Cada chamada gera tokens novos, impossível rastrear)
    
    Returns:
        PIIMasker configurado para LGPD
    """
    return PIIMasker(
        strategy="TOKEN",     # Tokens genéricos
        enable_ner=True,      # Usa IA para nomes
    )


# ============================================================================
# EXEMPLO DE USO
# ============================================================================
# Este código roda quando você executa o arquivo diretamente:
# python pii_masker.py
# ============================================================================

if __name__ == "__main__":
    # Texto de exemplo com vários tipos de PII
    sample_text = """
    Paciente Maria Silva, CPF 123.456.789-00, nascida em 15/03/1985.
    Telefone para contato: (11) 98765-4321.
    Endereço: Rua das Flores, 123 - São Paulo.
    Email: maria.silva@email.com
    
    Queixa principal: Dor de cabeça há 3 dias, com piora noturna.
    """
    
    # Cria o mascarador com configuração LGPD
    masker = create_lgpd_masker()
    
    # Executa o mascaramento
    result = masker.mask(sample_text)
    
    # Exibe resultados
    print("=" * 60)
    print("TEXTO ORIGINAL")
    print("=" * 60)
    print(result.original_text)
    
    print("\n" + "=" * 60)
    print("TEXTO MASCARADO")
    print("=" * 60)
    print(result.masked_text)
    
    print(f"\n{'=' * 60}")
    print(f"ENTIDADES DETECTADAS ({result.entity_count})")
    print("=" * 60)
    for entity in result.entities_found:
        print(f"  ✓ {entity.entity_type}: '{entity.original_value}' → '{entity.masked_value}'")
