import type { ModelId } from "@/lib/registry";
import {
  Brain,
  TreePine,
  BarChart3,
  GitBranch,
  Layers,
  BookOpen,
  RefreshCw,
  SlidersHorizontal,
} from "lucide-react";

interface Props {
  modelId: ModelId;
}

const explanations: Record<
  ModelId,
  {
    icon: React.ReactNode;
    what: string;
    train: string;
    learn: string;
    retrain: string;
    tuner: string;
  }
> = {
  rl: {
    icon: <Brain className="h-5 w-5" />,
    what:
      "Agente de Q-Learning tabular que aprende en tiempo real a comprar, vender o mantener. El estado se construye a partir de 5 indicadores (momentum, RSI, volatilidad, tendencia EMA y exposición actual) discretizados en 3 bins cada uno, generando ~243 estados posibles. No hay red neuronal: es una tabla Q (Map) que almacena el valor esperado de cada acción en cada estado.",
    train:
      "Durante el backtest inicial el agente explora con política ε-greedy. Cada vez que actúa, recibe una recompensa igual al cambio porcentual de equity tras la acción. Los valores Q se inicializan en [1,1,1] (comprar, mantener, vender) y se actualizan con descuento γ=0.92.",
    learn:
      "Usa tres mecanismos de aprendizaje simultáneos: (1) Eligibility traces — un buffer circular de 12 pasos recientes actualiza todos los estados visitados con influencia decreciente (γλ)^edad; (2) Experience replay — buffer de 100 transiciones (estado, acción, recompensa, siguiente estado), samplea 3 aleatorias cada 4 updates para evitar olvido catastrófico; (3) Decaimiento de ε — la exploración baja 0.15% por trade hasta un mínimo de 3%.",
    retrain:
      "No hay 'reentrenamiento' por lotes. El agente aprende online continuo: en cada tick de mercado nuevo observa el estado, actúa, recibe recompensa y actualiza Q. Esto significa que su comportamiento evoluciona permanentemente sin necesidad de fases de entrenamiento explícitas.",
    tuner:
      "El 'Tuner Online' cambia tres hiperparámetros en vivo: ε (exploración) va de 5% (conservador) a 35% (temerario); α (tasa de aprendizaje) de 0.06 a 0.16; γ (horizonte temporal) de 0.98 a 0.88. Un perfil agresivo explora más y aprende más rápido pero con mayor ruido; conservador planifica a más largo plazo.",
  },
  xgb: {
    icon: <TreePine className="h-5 w-5" />,
    what:
      "Gradient Boosting para clasificación binaria (¿subirá el precio en las próximas 5 barras?). Construye un ensamble de árboles de decisión de regresión donde cada nuevo árbol corrige los errores (gradientes negativos) del ensamble anterior. La salida es una probabilidad logística P(subida) obtenida de la suma ponderada de predicciones de todos los árboles.",
    train:
      "Entrenamiento inicial sobre las últimas ~400 barras históricas. Para cada árbol: (1) calcula gradientes e hessianos de la log-loss sobre las predicciones actuales; (2) construye un árbol de regresión de profundidad 3 que minimiza (g²)/(h+λ) en cada split; (3) actualiza los scores sumando learningRate × predicción del árbol. Usa 80% de las filas aleatorias por árbol (subsample) para regularizar.",
    learn:
      "El 'aprendizaje' ocurre durante la construcción de cada árbol: el algoritmo busca la feature y el umbral que maximizan la reducción de loss (gain). Tras cada árbol se recalcula la loss total; si no mejora >0.001 durante 5 árboles consecutivos (y ya hay ≥10 árboles), se detiene el entrenamiento (early stopping).",
    retrain:
      "Cada 25 ticks (≈25 min) se ejecuta retrainIncremental: añade 4 árboles nuevos sobre los últimos ~400 datos. El learning rate decae un 3% por reentrenamiento (nunca baja del 20% del valor base) para estabilizar el modelo conforme crece el ensamble. La base de árboles previos se conserva y se extiende.",
    tuner:
      "Cambia la complejidad y la agresividad del modelo: maxTrees (30–60), maxDepth (2–4), learningRate (0.05–0.15), lambda de regularización (0.7–1.5) y los umbrales de compra/venta (spread de 0.07–0.12). Agresivo = más árboles profundos y umbrales laxos; conservador = menos árboles, regularización alta y decisiones más estrictas.",
  },
  stat: {
    icon: <BarChart3 className="h-5 w-5" />,
    what:
      "Modelo estadístico puro que toma decisiones basadas en un score compuesto de 4 pruebas de hipótesis: momentum (Z-score del retorno), mean reversion (desviación de la media móvil), fuerza de tendencia (R² de regresión lineal sobre EMAs) y penalización por volatilidad. No usa machine learning: todo son tests estadísticos con parámetros adaptativos.",
    train:
      "No tiene fase de entrenamiento tradicional. En su lugar mantiene una ventana rodante de 200 observaciones con el algoritmo de Welford (mean y M2 acumulativo), calculando media y desviación estándar en tiempo O(1) por tick. Esto le permite adaptarse a cambios de régimen del mercado sin reentrenar.",
    learn:
      "La 'adaptación' ocurre vía un offset dinámico del umbral de decisión. Tras cada predicción, el modelo registra si acertó (el precio subió tras señal de compra). Cada 20 predicciones calcula el hit rate: si acierta >50% baja el umbral (más agresivo); si acierta <50% lo sube (más conservador). El ajuste es (hitRate−0.5)×0.15.",
    retrain:
      "No requiere reentrenamiento por lotes. La ventana Welford se actualiza automáticamente en cada nuevo tick (descarta la observación más antigua si supera 200). Los umbrales adaptativos también se ajustan online. Es el modelo más ligero computacionalmente.",
    tuner:
      "Modifica los parámetros de las 4 pruebas estadísticas: Z-momentum (1.5→0.7), Z-mean-reversion (1.8→1.0), R² mínimo de tendencia (0.10→0.02) y penalización por volatilidad (0.7→1.0). Conservador exige señales más extremas y tendencias más claras; agresivo acepta señales más débiles.",
  },
  rf: {
    icon: <GitBranch className="h-5 w-5" />,
    what:
      "Random Forest: ensamble de 60 árboles de clasificación (Gini) entrenados con bootstrap sampling. Cada árbol ve una muestra bootstrap (~63% de filas únicas) y en cada split solo evalúa un subconjunto aleatorio de 3–6 features. La predicción final es la media de probabilidades de todos los árboles.",
    train:
      "Entrenamiento inicial: para cada uno de los 60 árboles se genera una muestra bootstrap de tamaño N con reemplazo desde el dataset de ~400 barras. Las filas no seleccionadas (~37%) forman el conjunto OOB (Out-of-Bag). Tras entrenar cada árbol, se mide la precisión OOB. Las features usadas en splits se acumulan para calcular importancia de features normalizada.",
    learn:
      "Cada árbol aprende dividiendo recursivamente por la feature y umbral que maximizan la reducción de impureza Gini. El valor hoja es 1 si la mayoría de muestras en ese nodo son 'subida', 0 en caso contrario. El ensamble reduce varianza promediando muchos árboles correlacionados débilmente.",
    retrain:
      "Cada 25 ticks se añaden 6 árboles nuevos vía addTrees(). Cada nuevo árbol usa bootstrap sampling y se evalúa contra el OOB del ensamble completo. Si la precisión OOB no mejora >0.5% durante 3 árboles consecutivos, se detiene (early stopping). Hay un hard cap de nEstimators árboles totales.",
    tuner:
      "Cambia nEstimators (40–100 árboles), maxDepth (3–6), nFeatSample (2–6 features evaluadas por split) y los umbrales de compra/venta. Agresivo = más árboles profundos que pueden overfittear ligeramente a corto plazo; conservador = árboles más simples y generalistas.",
  },
  gru: {
    icon: <Layers className="h-5 w-5" />,
    what:
      "Red neuronal recurrente con células GRU (Gated Recurrent Unit). En cada paso procesa un vector de 10 features técnicas y actualiza un estado oculto mediante dos puertas sigmoides: update gate (z) que decide qué parte del estado anterior conservar, y reset gate (r) que decide qué olvidar antes de calcular el nuevo candidato. La salida es una probabilidad logística P(subida) obtenida de una capa densa sobre el último estado oculto, con Layer Normalization en cada paso temporal.",
    train:
      "Inicialización Xavier/Glorot para matrices Wz, Wr, Wn, Uz, Ur, Un y Wout. Optimizador Adam (β1=0.9, β2=0.999) con weight decay 1e-5. Entrenamiento por batches: samplea una ventana aleatoria de la serie y aplica BPTT completo sobre toda la secuencia de 20 pasos. Dropout 0.2 sobre el estado oculto durante el entrenamiento. Se entrena sobre las últimas ~400 barras.",
    learn:
      "BPTT completo propaga el gradiente desde la salida hacia atrás a través de todas las capas GRU y la LayerNorm por paso temporal. Usa gradient clipping dinámico: un EMA de la norma del gradiente ajusta el threshold de clipping. Además se entrenan los parámetros de LayerNorm (γ, β) junto con los pesos de las puertas.",
    retrain:
      "Cada 50 ticks (≈50 min) se reentrena con 20 pasos de BPTT sobre los datos más recientes. Los pesos se actualizan directamente, por lo que el modelo adapta gradualmente su comportamiento a los nuevos patrones del mercado sin necesidad de añadir capas.",
    tuner:
      "Cambia el tamaño del estado oculto (10–18 neuronas), la tasa de aprendizaje (0.002–0.008), la longitud de secuencia (12–20 pasos) y los umbrales de señal. Agresivo = red más grande que captura más patrones; conservador = red más pequeña y secuencias más cortas para generalizar mejor.",
  },
};

export function ModelExplanation({ modelId }: Props) {
  const e = explanations[modelId];
  const items = [
    { label: "Qué hace", text: e.what, icon: <BookOpen className="h-4 w-4" /> },
    { label: "Cómo se entrena", text: e.train, icon: <Brain className="h-4 w-4" /> },
    { label: "Cómo aprende", text: e.learn, icon: <RefreshCw className="h-4 w-4" /> },
    { label: "Cómo se reentrena", text: e.retrain, icon: <RefreshCw className="h-4 w-4" /> },
    { label: "Efecto del Tuner Online", text: e.tuner, icon: <SlidersHorizontal className="h-4 w-4" /> },
  ];

  return (
    <section className="glass-card rounded-2xl p-5">
      <div className="mb-4 flex items-center gap-2">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-secondary text-muted-foreground">
          {e.icon}
        </div>
        <h3 className="text-sm font-bold tracking-tight">Cómo funciona este modelo</h3>
      </div>
      <div className="space-y-4">
        {items.map((item) => (
          <div key={item.label} className="rounded-lg border border-border/40 bg-secondary/30 p-3">
            <div className="mb-1.5 flex items-center gap-1.5 text-[11px] font-semibold text-muted-foreground">
              {item.icon}
              <span>{item.label}</span>
            </div>
            <p className="text-[11px] leading-relaxed text-foreground/80">{item.text}</p>
          </div>
        ))}
      </div>
    </section>
  );
}
