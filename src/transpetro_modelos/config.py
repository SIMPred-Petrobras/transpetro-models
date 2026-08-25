from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class EquipmentConfig:
    equipment_id: str
    failure_date: datetime
    failure_description: str
    dataset_name: str
    datetime_column: Optional[str]  # "Data Hora" para B-402E, "Timestamp" para B-4064A novos, None para demais
    exclusion_days_before: int
    preprocessing_steps: list[dict]
    pre_split_steps: list[dict] = field(default_factory=list)  # resample, filter_running (roda antes do split)
    preprocess_presets: dict[str, list[dict]] = field(default_factory=dict)
    local_feather: Optional[str] = None  # override path for local loading (relative to project root)
    dataset_file_stem: Optional[str] = None  # override filename stem when fetching from ClearML (default: equipment_id)
    val_start_date: Optional[datetime] = None  # fixed validation start date (e.g., Jul 1)
    val_end_date: Optional[datetime] = None  # fixed validation end (branch Lara); rows between it and the cutoff are excluded
    # Janelas de avaliação (failure_detection_metrics). None = usa o default global do CLI (30 / 60).
    # Configurar por equipamento quando a série é curta ou o sinal de falha é concentrado:
    # normal_end_days precisa cair DENTRO dos dados, senão a janela normal fica vazia,
    # normal_alert_rate=0 e a constraint --max-fp-rate é silenciosamente desativada.
    prefailure_days: Optional[int] = None
    normal_end_days: Optional[int] = None


_THERMAL_VIB_FEATURES = [
    "Temperatura Bomba LNA",
    "Vibração Bomba LNA",
    "Temperatura Motor LNA",
]

B4064A_NOVOS_PREPROCESS_PRESETS: dict[str, list[dict]] = {
    "baseline": [
        {"step": "clip"},
        {"step": "normalize", "method": "robust"},
    ],
    "moving_average": [
        {"step": "moving_average", "window": 3, "min_periods": 1},
        {"step": "clip"},
        {"step": "normalize", "method": "robust"},
    ],
    "knn": [
        {"step": "knn_impute", "n_neighbors": 3, "weights": "distance"},
        {"step": "clip"},
        {"step": "normalize", "method": "robust"},
    ],
    "moving_average_knn": [
        {"step": "knn_impute", "n_neighbors": 3, "weights": "distance"},
        {"step": "moving_average", "window": 3, "min_periods": 1},
        {"step": "clip"},
        {"step": "normalize", "method": "robust"},
    ],
    # Feature-selected presets: only the sensors that best capture bearing degradation
    "thermal": [
        {"step": "select_features", "features": _THERMAL_VIB_FEATURES},
        {"step": "clip"},
        {"step": "normalize", "method": "robust"},
    ],
    "thermal_ma": [
        {"step": "select_features", "features": _THERMAL_VIB_FEATURES},
        {"step": "moving_average", "window": 3, "min_periods": 1},
        {"step": "clip"},
        {"step": "normalize", "method": "robust"},
    ],
    # Presets com features de janela deslizante (std, média, diff).
    # Dados horários: windows=[6, 24] = 6 h e 24 h.
    # O add_rolling_features vem antes do clip para que os limites sejam
    # calculados sobre o conjunto enriquecido.
    "rolling": [
        {"step": "add_rolling_features", "windows": [6, 24]},
        {"step": "clip"},
        {"step": "normalize", "method": "robust"},
    ],
    "rolling_ma": [
        {"step": "moving_average", "window": 3, "min_periods": 1},
        {"step": "add_rolling_features", "windows": [6, 24]},
        {"step": "clip"},
        {"step": "normalize", "method": "robust"},
    ],
    "rolling_thermal": [
        {"step": "select_features", "features": _THERMAL_VIB_FEATURES},
        {"step": "add_rolling_features", "windows": [6, 24]},
        {"step": "clip"},
        {"step": "normalize", "method": "robust"},
    ],
}


# Presets para bases interpoladas (branch Lara): normalização standard, variantes knn/MA.
_INTERPOLATED_PRESETS_LARA: dict[str, list[dict]] = {
    "baseline_interpolated": [
        {"step": "clip"},
        {"step": "normalize", "method": "standard"},
    ],
    
    "moving_average_interpolated": [
        {"step": "moving_average", "window": 3, "min_periods": 1},
        {"step": "clip"},
        {"step": "normalize", "method": "standard"},
    ],

    "knn_interpolated": [
        {"step": "knn_impute", "n_neighbors": 3, "weights": "distance"},
        {"step": "clip"},
        {"step": "normalize", "method": "standard"},
    ]
}


EQUIPMENT_CONFIGS: dict[str, EquipmentConfig] = {
    # Base interpolada do B-4064A (versão branch Lara: Timestamp, val_end_date e presets interpolados)
    "B-4064A_interpolated": EquipmentConfig(
        equipment_id="B-4064A_interpolated",
        failure_date=datetime(2024, 8, 30, 7, 58),
        failure_description="Roçamento interno do rotor com a carcaça da bomba",
        dataset_name="transpetro-b-4064a_interpolated",
        datetime_column="Timestamp",
        exclusion_days_before=10,
        local_feather="Dados/B-4064A_interpolated.csv",
        val_start_date=datetime(2024, 8, 11),
        val_end_date=datetime(2024, 8, 20),
        pre_split_steps=[
            {"step": "filter_running", "column": "Corrente", "threshold": 30},
            {"step": "remove_transients", "minutes": 10},
        ],
        preprocessing_steps=deepcopy(_INTERPOLATED_PRESETS_LARA["baseline_interpolated"]),
        preprocess_presets=deepcopy(_INTERPOLATED_PRESETS_LARA),
    ),
    "B-4064A-novos": EquipmentConfig(
        equipment_id="B-4064A-novos",
        failure_date=datetime(2024, 8, 30, 7, 58),
        failure_description="Roçamento interno do rotor com a carcaça da bomba",
        dataset_name="transpetro-b-4064a-novos",
        datetime_column="Timestamp",
        exclusion_days_before=10,
        local_feather="Dados-novos/B-4064A_novos.feather",
        val_start_date=datetime(2024, 7, 1),
        pre_split_steps=[
            {"step": "remove_sensor_errors", "error_values": [-25.0]},
            {"step": "resample", "freq": "1h"},
            {"step": "ffill", "limit": 6},
            {"step": "filter_running", "column": "Corrente", "threshold": 5.0},
            {"step": "filter_running", "column": "Pressão Descarga", "threshold": 0.0},
            # Remove the first 2 h after each restart (warm-up period gives anomalous low temps).
            # gap_minutes=90 avoids treating every 1-h interval as a restart.
            {"step": "remove_transients", "minutes": 120, "gap_minutes": 90},
        ],
        preprocessing_steps=deepcopy(B4064A_NOVOS_PREPROCESS_PRESETS["baseline"]),
        preprocess_presets=deepcopy(B4064A_NOVOS_PREPROCESS_PRESETS),
    ),
    # Config "produção" do B-4064A com os ajustes da análise de regime (docs/analise_b4064a_regime.md):
    # mascara dropouts de vibração e adiciona o preset load_residual (resíduo Temp~Corrente,
    # invariante ao regime de carga). Reusa o MESMO dataset do ClearML (B-4064A-novos).
    "B-4064A-prod": EquipmentConfig(
        equipment_id="B-4064A-prod",
        failure_date=datetime(2024, 8, 30, 7, 58),
        failure_description="Roçamento interno do rotor com a carcaça da bomba",
        dataset_name="transpetro-b-4064a-novos",
        dataset_file_stem="B-4064A-novos",  # arquivo dentro do dataset ClearML transpetro-b-4064a-novos
        datetime_column="Timestamp",
        exclusion_days_before=10,
        local_feather="Dados-novos/B-4064A_novos.feather",
        val_start_date=datetime(2024, 7, 1),
        # Assinatura da falha de 2024 é nos últimos ~2 dias → janela pré-falha curta (3d casa
        # com o precursor; 5d diluía); janela normal com buffer (jan-ago/2024 dá normal cheio).
        prefailure_days=3,
        normal_end_days=20,
        pre_split_steps=[
            {"step": "remove_sensor_errors", "error_values": [-25.0]},
            {"step": "resample", "freq": "1h"},
            {"step": "ffill", "limit": 6},
            {"step": "filter_running", "column": "Corrente", "threshold": 5.0},
            {"step": "filter_running", "column": "Pressão Descarga", "threshold": 0.0},
            # Mascara dropouts do sensor de vibração (<=0.2, inclui negativos impossíveis).
            {"step": "filter_running", "column": "Vibração Bomba LNA", "threshold": 0.2},
            {"step": "remove_transients", "minutes": 120, "gap_minutes": 90},
        ],
        preprocessing_steps=[
            {"step": "clip"},
            {"step": "normalize", "method": "robust"},
        ],
        preprocess_presets={
            "baseline": [
                {"step": "clip"},
                {"step": "normalize", "method": "robust"},
            ],
            # Resíduo de temperatura do mancal condicionado à carga (substitui a temp absoluta):
            # captura "mais quente do que a carga explica" e é invariante à troca de regime de carga.
            "load_residual": [
                {"step": "add_load_residual",
                 "temp_columns": ["Temperatura Bomba LA", "Temperatura Bomba LNA"],
                 "load_column": "Corrente", "replace": True},
                {"step": "clip"},
                {"step": "normalize", "method": "robust"},
            ],
            # Resíduo + média móvel curta (suaviza ruído do resíduo antes do clip/normalize).
            "load_residual_ma": [
                {"step": "add_load_residual",
                 "temp_columns": ["Temperatura Bomba LA", "Temperatura Bomba LNA"],
                 "load_column": "Corrente", "replace": True},
                {"step": "moving_average", "window": 3, "min_periods": 1},
                {"step": "clip"},
                {"step": "normalize", "method": "robust"},
            ],
        },
    ),
    "B-8802B": EquipmentConfig(
        equipment_id="B-8802B",
        failure_date=datetime(2022, 7, 6, 10, 0),
        failure_description="Trinca nas lâminas do acoplamento",
        dataset_name="transpetro-b-8802b",
        datetime_column=None,
        exclusion_days_before=10,
        # Falha súbita (trinca no acoplamento): o sinal de vibração só emerge na última
        # semana (precursor VibLNA ~01/jul, rampa sustentada ~04/jul). prefailure_days=30
        # diluía o alvo com 3 semanas de operação normal. normal_end_days=60 colocava o fim
        # do período normal em 07/mai — ANTES do início dos dados (15/mai) — deixando a janela
        # normal vazia e desativando a constraint --max-fp-rate. 7/20 concentra o alvo na
        # semana de degradação e mede FP sobre ~5.2k amostras normais limpas (gap de 13 dias).
        prefailure_days=7,
        normal_end_days=20,
        pre_split_steps=[
            {"step": "remove_sensor_errors", "error_values": [0.0]},
            {"step": "filter_running", "column": "Pressão Descarga", "threshold": 35.0},
            {"step": "resample", "freq": "5min"},
            {"step": "ffill", "limit": 4},
            # Temperatura leva ~90 min para estabilizar após reinício (gap_minutes=30 evita
            # tratar cada intervalo de 5min como reinício independente).
            {"step": "remove_transients", "minutes": 90, "gap_minutes": 30},
            {"step": "select_features", "features": ["Pressão Sucção", "Pressão Descarga", "Vibração Bomba LA", "Vibração Bomba LNA", "Temperatura Bomba LA"]},
        ],
        preprocessing_steps=[
            {"step": "clip", "upper_pct": 99.9},
            {"step": "normalize", "method": "robust"},
        ],
        # Dados em 5 min: windows=[12, 72] = 1 h e 6 h.
        # Preset "rolling" removido: para falha súbita as features std/diff não melhoram a
        # separabilidade do sinal (que vive no degrau bruto de vibração) e diluem o sinal
        # multivariado, além de serem as mais propensas a falso positivo no período normal.
        # rolling_ma (suaviza antes) é preferível se um preset com janelas for desejado.
        preprocess_presets={
            "baseline": [
                {"step": "clip", "upper_pct": 99.9},
                {"step": "normalize", "method": "robust"},
            ],
            "rolling_ma": [
                {"step": "moving_average", "window": 3, "min_periods": 1},
                {"step": "add_rolling_features", "windows": [12, 72]},
                {"step": "clip", "upper_pct": 99.9},
                {"step": "normalize", "method": "robust"},
            ],
        },
    ),
    "B-8802B-2025": EquipmentConfig(
        equipment_id="B-8802B-2025",
        # RETREINO PÓS-DRIFT (ago/2026): o modelo de 2022 dá 12% de alarme em 2025-26
        # (reparo pós-falha + faixa de regimes mais ampla que as 6 semanas do treino original).
        # NÃO há falha conhecida neste período: failure_date abaixo é SENTINELA (fim dos dados
        # 2026-08-10 + 1 dia) só para satisfazer o split; a janela pré-falha resultante mede FP
        # em dado recente, não detecção. Seleção deve usar --select-by heldout (FP em 2026
        # nunca visto). Sensibilidade é validada à parte, pontuando a falha de 2022 com o
        # modelo novo. Dados: b8802b-2025-2026/ (COV IFIX) -> grade 1 min hold-last-value.
        failure_date=datetime(2026, 8, 11),
        failure_description="sem falha no período (retreino pós-drift; reparo após a falha de 2022)",
        dataset_name="transpetro-b-8802b-2025",
        datetime_column=None,
        exclusion_days_before=1,
        prefailure_days=7,
        normal_end_days=20,
        # treino = 2025 inteiro (cobre os regimes); held-out = jan-jun/2026
        val_start_date=datetime(2026, 1, 1),
        val_end_date=datetime(2026, 6, 1),
        local_feather="Dados-novos/B-8802B-2025.feather",
        pre_split_steps=[
            {"step": "remove_sensor_errors", "error_values": [0.0]},
            {"step": "filter_running", "column": "Pressão Descarga", "threshold": 35.0},
            {"step": "resample", "freq": "5min"},
            {"step": "ffill", "limit": 4},
            {"step": "remove_transients", "minutes": 90, "gap_minutes": 30},
            # Máscara de transiente de PROCESSO (manobra): degrau >1,5×p99 da variação normal em
            # 15 min (sucção 2,4 bar / descarga 4,8 bar) -> ignora os 90 min seguintes. Corta os
            # blips de FP dirigidos por pressão sem alterar a sensibilidade (validado: falha 2022 e
            # falha sintética inalteradas; FP held-out 0,063% -> 0,038%).
            {"step": "remove_regime_transients", "columns": ["Pressão Sucção", "Pressão Descarga"],
             "deltas": [2.4, 4.8], "minutes": 90, "window": 3},
            {"step": "select_features", "features": ["Pressão Sucção", "Pressão Descarga", "Vibração Bomba LA", "Vibração Bomba LNA", "Temperatura Bomba LA"]},
        ],
        preprocessing_steps=[
            {"step": "clip", "upper_pct": 99.9},
            {"step": "normalize", "method": "robust"},
        ],
        preprocess_presets={
            "baseline": [
                {"step": "clip", "upper_pct": 99.9},
                {"step": "normalize", "method": "robust"},
            ],
            "rolling_ma": [
                {"step": "moving_average", "window": 3, "min_periods": 1},
                {"step": "add_rolling_features", "windows": [12, 72]},
                {"step": "clip", "upper_pct": 99.9},
                {"step": "normalize", "method": "robust"},
            ],
        },
    ),
    "B-8802B-8s": EquipmentConfig(
        equipment_id="B-8802B-8s",
        failure_date=datetime(2022, 7, 6, 10, 0),
        failure_description="Trinca nas lâminas do acoplamento",
        dataset_name="transpetro-b-8802b",
        datetime_column=None,
        exclusion_days_before=10,
        dataset_file_stem="B-8802B",
        pre_split_steps=[
            {"step": "remove_sensor_errors", "error_values": [0.0]},
            {"step": "filter_running", "column": "Pressão Descarga", "threshold": 35.0},
            {"step": "remove_transients", "minutes": 15},
            {"step": "resample", "freq": "5min"},
            {"step": "ffill", "limit": 4},
        ],
        preprocessing_steps=[
            {"step": "clip", "upper_pct": 99.9},
            {"step": "normalize", "method": "robust"},
        ],
    ),
    "B-8802B-8s-nfr": EquipmentConfig(
        equipment_id="B-8802B-8s-nfr",
        failure_date=datetime(2022, 7, 6, 10, 0),
        failure_description="Trinca nas lâminas do acoplamento",
        dataset_name="transpetro-b-8802b",
        datetime_column=None,
        exclusion_days_before=10,
        dataset_file_stem="B-8802B",
        pre_split_steps=[
            {"step": "remove_sensor_errors", "error_values": [0.0]},
            {"step": "remove_transients", "minutes": 15},
            {"step": "resample", "freq": "5min"},
            {"step": "ffill", "limit": 4},
        ],
        preprocessing_steps=[
            {"step": "clip", "upper_pct": 99.9},
            {"step": "normalize", "method": "robust"},
        ],
    ),
    "B-6511502A": EquipmentConfig(
        equipment_id="B-6511502A",
        failure_date=datetime(2023, 5, 15, 0, 0),
        failure_description="Quebra das lâminas do acoplamento",
        dataset_name="transpetro-b-6511502a",
        datetime_column=None,
        exclusion_days_before=10,
        # Falha súbita (acoplamento): sinal só no último dia; dados acabam ~1,4d antes da falha.
        # prefailure curto (3d) casa com o sinal; normal_end=20 dá janela normal cheia.
        prefailure_days=3,
        normal_end_days=20,
        local_feather="DadosV2/B-6511502A_pivoted.feather",
        pre_split_steps=[
            {"step": "remove_sensor_errors", "error_values": [32767.0]},
            {"step": "filter_running", "column": "CORRENTE ELÉTRICA DO MOTOR", "threshold": 60.0},
            {"step": "remove_transients", "minutes": 15},
            {"step": "resample", "freq": "5min"},
            {"step": "ffill", "limit": 4},
            {"step": "select_features", "features": [
                "CORRENTE ELÉTRICA DO MOTOR",
                "PRESSÃO SUCÇÃO",
                "PRESSÃO DESCARGA",
                "DESLOC. AXIAL EIXO BB LNA 1 ZE-50",
                "DESLOC. AXIAL EIXO BB LNA 2 ZE-51",
                "TEMP. MANCAL LNA MOT TE-07A1/A2",
                "VIB. MANCAL RADIAL BB LA 0° VE-50C",
                "VIB. MANCAL RADIAL BB LA 90° VE-51C",
                "VIB. MANCAL RADIAL BB LNA 0° VE-50D",
                "VIB. MANCAL RADIAL BB LNA 90° VE-51D",
                "VIB. MANCAL RADIAL MOT LA 0° VE-51B",
                "VIB. MANCAL RADIAL MOT LA 90° VE-50B",
                "VIB. MANCAL RADIAL MOT LNA 0° VE-50A",
                "VIB. MANCAL RADIAL MOT LNA 90° VE-51A",
            ]},
        ],
        preprocessing_steps=[
            {"step": "clip", "upper_pct": 99.9},
            {"step": "normalize", "method": "robust"},
        ],
    ),
    "B-4703.24001B": EquipmentConfig(
        equipment_id="B-4703.24001B",
        failure_date=datetime(2022, 10, 1, 0, 0),
        failure_description="Desgaste mancal LNA motor elétrico",
        dataset_name="transpetro-b-4703-24001b",
        datetime_column="Data Hora",
        exclusion_days_before=10,
        local_feather="DadosV2/B-4703.24001B.feather",
        pre_split_steps=[
            {"step": "filter_running", "column": "Corrente", "threshold": 5.0},
            {"step": "remove_transients", "minutes": 15},
            {"step": "resample", "freq": "5min"},
            {"step": "ffill", "limit": 4},
            {"step": "select_features", "features": [
                "Corrente",
                "Pressão Sucção",
                "Pressão Decsrga",
                "Vibração Bomba LA",
                "Vibração Bomba LNA",
                "Vibração Motor LA X",
                "Vibração Motor LA Y",
                "Vibração Motor LNA X",
                "Vibração Motor LNA Y",
                "Temperatura Bomba LA",
                "Temperatura Bomba LNA",
                "Temperatura Motor LA",
                "Temperatura Motor LNA",
            ]},
        ],
        preprocessing_steps=[
            {"step": "clip", "upper_pct": 99.9},
            {"step": "normalize", "method": "robust"},
        ],
    ),
    "B-0302C": EquipmentConfig(
        equipment_id="B-0302C",
        failure_date=datetime(2024, 8, 30, 0, 0),
        failure_description="Falha no motor elétrico com aumento de vibração",
        dataset_name="transpetro-b-0302c",
        datetime_column=None,
        exclusion_days_before=10,
        local_feather="DadosV2/B-0302C_pivoted.feather",
        pre_split_steps=[
            # Sensores elétricos (temperatura enrolamento, desbalanço, tensão) estão todos
            # zerados no dataset — apenas vibração e pressão diferencial têm dados contínuos.
            # select_features antes do remove_sensor_errors garante que só as colunas úteis
            # sejam verificadas (remove as 83 linhas de falha de telemetria).
            {"step": "select_features", "features": [
                "Vibração Mancal Bomba LA",
                "Pressão diferencial filtro",
            ]},
            {"step": "remove_sensor_errors", "error_values": [0.0]},
            {"step": "resample", "freq": "5min"},
            {"step": "ffill", "limit": 4},
        ],
        preprocessing_steps=[
            {"step": "clip", "upper_pct": 99.9},
            {"step": "normalize", "method": "robust"},
        ],
    ),
    "B-90001A": EquipmentConfig(
        equipment_id="B-90001A",
        failure_date=datetime(2021, 8, 28, 0, 0),
        failure_description="Afrouxamento no aperto dos parafusos do mancal do lado acoplado da bomba",
        dataset_name="transpetro-b-90001a",
        datetime_column=None,
        exclusion_days_before=10,
        preprocessing_steps=[
            {"step": "normalize", "method": "standard"},
        ],
    ),
    # --- Equipamentos FRACOS (triagem det×FP) — configurados para DOCUMENTAR o resultado ruim ---
    # B-402E: quebra de barra do rotor + colisão estator (falha catastrófica/súbita com TRIP).
    # Triagem: ~5% @1%FP (assinatura = transiente de partida; sem precursor gradual). Filtro Corrente>200.
    "B-402E": EquipmentConfig(
        equipment_id="B-402E",
        failure_date=datetime(2019, 10, 30, 11, 6),
        failure_description="Quebra de barra do rotor do motor com colisão no enrolamento estatórico (TRIP)",
        dataset_name="transpetro-b-402e",
        datetime_column="Data Hora",
        exclusion_days_before=10,
        local_feather="Dados/B-402E.feather",
        prefailure_days=7,
        normal_end_days=20,
        pre_split_steps=[
            {"step": "resample", "freq": "5min"},
            {"step": "ffill", "limit": 4},
            {"step": "filter_running", "column": "Corrente", "threshold": 200.0},
            {"step": "remove_transients", "minutes": 30, "gap_minutes": 15},
            {"step": "select_features", "features": [
                "Corrente", "Vazão", "Vibração Bomba LA",
                "Temperatura Estator U", "Temperatura Estator V",
                "Temperatura Estator Wa", "Temperatura Estator Wb",
                "Temperatura Mancal LA Motor", "Temperatura Mancal LNA Motor",
            ]},
        ],
        preprocessing_steps=[
            {"step": "clip", "upper_pct": 99.9},
            {"step": "normalize", "method": "robust"},
        ],
    ),
    # B-5401A: motor em curto (1ª falha 2024-08-10). Triagem: ~0% @1%FP (curto súbito sem precursor).
    # Sensor de vibração MORTO (zeros) e Temp Mancal Motor LA SATURADA(100) — excluídos do select.
    "B-5401A": EquipmentConfig(
        equipment_id="B-5401A",
        failure_date=datetime(2024, 8, 10, 0, 0),
        failure_description="Motor elétrico em curto (substituído)",
        dataset_name="transpetro-b-5401a",
        datetime_column=None,
        exclusion_days_before=10,
        local_feather="DadosV2/B-5401A_pivoted.feather",
        prefailure_days=7,
        normal_end_days=20,
        pre_split_steps=[
            {"step": "filter_running", "column": "Corrente", "threshold": 50.0},
            {"step": "resample", "freq": "5min"},
            {"step": "ffill", "limit": 4},
            {"step": "remove_transients", "minutes": 30, "gap_minutes": 15},
            {"step": "select_features", "features": [
                "Corrente", "Indicador de Velocidade", "Pressão de Descarga", "Pressão de Sucção",
                "Temperatura Mancal Bomba LA", "Temperatura Mancal Bomba LNA",
                "Temperatura Mancal Motor LNA",
            ]},
        ],
        preprocessing_steps=[
            {"step": "clip", "upper_pct": 99.9},
            {"step": "normalize", "method": "robust"},
        ],
    ),
    # --- Equipamentos da branch Lara (bases interpoladas; usam val_end_date) ---
    "B-3403C_interpolated": EquipmentConfig(
        equipment_id="B-3403C_interpolated",
        failure_date=datetime(2023, 9, 12),
        failure_description="Quebra da ponta do eixo LNA da bomba",
        dataset_name="transpetro-b-3403c_interpolated",
        datetime_column="Timestamp",
        exclusion_days_before=10,
        local_feather="Dados/B-3403C_interpolated.csv",
        val_start_date=datetime(2023, 8, 24),
        val_end_date=datetime(2023, 9, 2),
        pre_split_steps=[
            {"step": "filter_running", "column": "Corrente", "threshold": 1},
            {"step": "remove_transients", "minutes": 10},
        ],
        preprocessing_steps=deepcopy(_INTERPOLATED_PRESETS_LARA["baseline_interpolated"]),
        preprocess_presets=deepcopy(_INTERPOLATED_PRESETS_LARA),
    ),
    "B-90001A_interpolated": EquipmentConfig(
        equipment_id="B-90001A_interpolated",
        failure_date=datetime(2021, 8, 28),
        failure_description="Afrouxamento no aperto dos parafusos do mancal do lado acoplado da bomba",
        dataset_name="transpetro-b-90001a_interpolated",
        datetime_column="Timestamp",
        exclusion_days_before=10,
        local_feather="Dados/B-90001A_interpolated.csv",
        val_start_date=datetime(2021, 8, 9),
        val_end_date=datetime(2021, 8, 18),
        pre_split_steps=[
            {"step": "filter_threshold", "columns": [
                "Vibração Motor LNA Y", "Vibração Motor LA X", "Vibração Motor LA Y",
                "Vibração Bomba LA X", "Vibração Bomba LA Y",
                "Vibração Bomba LNA X", "Vibração Bomba LNA Y",
            ], "threshold": 10},
            {"step": "remove_transients", "minutes": 10},
        ],
        preprocessing_steps=deepcopy(_INTERPOLATED_PRESETS_LARA["baseline_interpolated"]),
        preprocess_presets=deepcopy(_INTERPOLATED_PRESETS_LARA),
    ),
}


def get_preprocessing_steps(equipment_id: str, preset: str = "baseline") -> list[dict]:
    config = EQUIPMENT_CONFIGS[equipment_id]
    if config.preprocess_presets:
        if preset not in config.preprocess_presets:
            available = ", ".join(sorted(config.preprocess_presets))
            raise ValueError(f"Unknown preprocess preset '{preset}' for {equipment_id}. Available: {available}")
        return deepcopy(config.preprocess_presets[preset])

    if preset != "baseline":
        raise ValueError(f"Equipment {equipment_id} only supports preprocess_preset='baseline'")

    return deepcopy(config.preprocessing_steps)
