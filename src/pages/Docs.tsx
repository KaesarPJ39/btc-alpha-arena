import { BrainCircuit, LineChart, Sigma, TreePine, Network, Activity, AlertTriangle } from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { MODEL_IDS, MODELS, type ModelId } from "@/lib/registry";

interface DocEntry {
  id: ModelId;
  icon: LucideIcon;
  title: string;
  subtitle: string;
  howItWorks: string;
  hyperparams: string[];
  aggressiveness: string;
  caveats: string;
}

const DOCS: DocEntry[] = [
  {
    id: "rl",
    icon: BrainCircuit,
    title: "Agente RL — Q-Learning tabular",
    subtitle: "Aprendizaje por refuerzo online ε-greedy con actualización incremental",
    howItWorks:
      "El estado se discretiza en 5 bins por cada una de 5 dimensiones (momentum, RSI, volatilidad, trend, posición). " +
      "La tabla Q se inicializa en 0 y se actualiza en cada paso con Q(s,a) ← Q(s,a) + α·[r + γ·max Q(s',·) − Q(s,a)]. " +
      "La recompensa es el cambio porcentual de equity escalado por 1000, y se moldea por riesgo " +
      "(penalización basada en la varianza reciente de las recompensas para favorecer estrategias con buen Sharpe-like). " +
      "La política ε-greedy explora con probabilidad ε, que decae geométricamente. Cada estado-action recibe un " +
      "α posterior adaptado al número de visitas del estado (más visitas → menor α).",
    hyperparams: [
      "α (tasa de aprendizaje): 0.06→0.16 según agresividad",
      "γ (factor de descuento): 0.88→0.98 (más agresivo = más a corto plazo)",
      "ε (exploración): parte de 0.25 y decae a 0.03 con factor 0.9985",
      "5 bins por dimensión → hasta 3125 estados posibles",
    ],
    aggressiveness:
      "A mayor agresividad: ε inicial más alto (más exploración), α mayor (aprende más rápido), γ menor " +
      "(descuenta más el futuro) y el sesgo de acción aleatoria se inclina a comprar. Conservador reduces " +
      "exploración y aumenta la paciencia del valor futuro.",
    caveats:
      "Q-Learning tabular no escala bien a estados continuos (pero el binning lo controla en BTC intradía). " +
      "Recompensa pura Δequity puede empujar al agente a operar en exceso, mitigado por cooldown de 30 s " +
      "y shaping por varianza.",
  },
  {
    id: "xgb",
    icon: LineChart,
    title: "XGBoost — Gradient Boosting para clasificación",
    subtitle: "Implementación propia de boosting con árboles de regresión y loss logística",
    howItWorks:
      "Entrena un ensemble secuencial de árboles poco profundos. Cada árbol se ajusta a los gradientes y " +
      "hessianos de la log-loss de la dirección del próximo retorno (1 = sube, 0 = baja). Capa de boosts: " +
      "p(t+1) = σ(baseScore + Σ η · tree_i(x)), con baseScore = log-odds de la clase positiva. " +
      "Cada iter construye un árbol buscando el split con mejor gain = gL²/(hL+λ) + gR²/(hR+λ) − g²/(h+λ). " +
      "Entrena una vez con datos históricos y luego reentrena incrementalmente añadiendo árboles cada 25 ticks.",
    hyperparams: [
      "maxTrees: 30→60 según agresividad",
      "maxDepth: 2→4 según agresividad",
      "learningRate (η): 0.05→0.15",
      "λ (regularización L2): 0.7→1.5",
      "minChildWeight: 8 (anti split ruidoso)",
      "Umbrales de señal: ±0.5±spread",
    ],
    aggressiveness:
      "A mayor agresividad: más árboles, más profundidad (más capacidad), learning rate mayor, menor λ " +
      "(menos regularización) y umbrales más cercanos a 0.5 (opera más). Conservador: parsimonia + mayor separación probabilística.",
    caveats:
      "Implementación propia sin GPU. Puede sobreajustar con features ruido; controlado por maxDepth, λ y " +
      "minChildWeight. La importancia de features es por frecuencia de split, no SHAP-like.",
  },
  {
    id: "stat",
    icon: Sigma,
    title: "Statistical — Pruebas de hipótesis compuestas",
    subtitle: "Combina Z-score de retorno, t-test de momentum, R² de tendencia y mean reversion",
    howItWorks:
      "No aprende de datos: aplica estadística descriptiva e inferencial. Calcula 5 pruebas estadísticas: " +
      "(1) z-score del último retorno vs media histórica incremental; (2) t-test direccional sobre momentum5; " +
      "(3) Z de mean reversion sobre RSI; (4) score de tendencia EMA21/EMA50 + precio/EMA50; " +
      "(5) alineación de racha (mom5 y mom15). Cada prueba emite un voto firmado. La suma se filtra por " +
      "volatilidad (mercado choppy → veto de operación) y se compara contra un umbral compuesto.",
    hyperparams: [
      "zMomentum: umbral 0.7→1.5 del estadístico t de momentum",
      "zMeanReversion: umbral 1.0→1.8 de Z del último retorno",
      "r2TrendMin: 0.02→0.10 de score mínimo de tendencia",
      "volatilityPenalty: 0.7→1.0 (más alto = más veto en volátil)",
      "Umbral compuesto: 1.2 - 0.7 · agresividad",
    ],
    aggressiveness:
      "Conservador sube los umbrales (menos operaciones, sólo señales muy claras) y aumenta el veto por volatilidad. " +
      "Agresivo baja los umbrales (más permeable a señales débiles) y relaja el veto.",
    caveats:
      "Es un modelo estático sin capacidad de aprender de errores pasados, pero robusto frente a ruido porque " +
      "sus parámetros se razonan estadísticamente. Útil como benchmark y como complemento a modelos ML.",
  },
  {
    id: "rf",
    icon: TreePine,
    title: "Random Forest — Bagging con árboles Gini",
    subtitle: "Ensemble de árboles independientes con bootstrap y muestreo aleatorio de features",
    howItWorks:
      "Construye nEstimators árboles de decisión independientes. Para cada árbol: (1) muestrea con reemplazo " +
      "el training set (bootstrap); (2) en cada split considera sólo una fracción aleatoria de features (sqrt por defecto) " +
      "para descorrelacionar árboles; (3) usa criterio de impureza Gini en lugar de boost de gradiente. " +
      "La predicción es la fracción de árboles que votan 'sube'. Calcula OOB (out-of-bag) estimate como " +
      "estimación no sesgada del accuracy del modelo. Mantenimiento: añade árboles incrementales cada tick batch.",
    hyperparams: [
      "n_estimators: 40→100 según agresividad",
      "max_depth: 3→6",
      "n_feat_sample: 2→6 (por split)",
      "min_child_split: 4",
      "Umbral buy: 0.5+spread, sell: 0.5−spread",
      "OOB estimate: accuracy estimado en muestras no usadas en bootstrap",
    ],
    aggressiveness:
      "A mayor agresividad: más árboles, más profundidad, más features por split (mayor capacidad), " +
      "umbrales más cercanos a 0.5. Conservador: 40 árboles poco profundos + umbrales anchos (sólo confiando).",
    caveats:
      "Random Forest es robusto pero no captura dinámica temporal (cada árbol i.i.d.). " +
      "Aproxima probabilidad por votación mayoritaria. Sufre de " +
      "correlación si las features técnicas contienen redundancias (RSI/MACD similares).",
  },
  {
    id: "gru",
    icon: Network,
    title: "GRU — Red Neuronal Recurrente",
    subtitle: "Célula GRU con puertas de actualización y reinicio · BPTT completo",
    howItWorks:
      "Red neuronal con células Gated Recurrent Unit. En cada paso procesa un vector de 10 features técnicas " +
      "y mantiene un estado interno oculto (h). Dos puertas sigmoides controlan qué se conserva (update gate z) " +
      "y qué se reinicia (reset gate r) antes de calcular el nuevo candidato. " +
      "Salida: σ(Wout·h + bout) = P(subida próxima barra). Entrena con BPTT completo sobre secuencias de 20 " +
      "pasos. Xavier/Glorot init. Loss = log-loss. " +
      "Adam optimizer con weight decay 1e-5 y LayerNorm en cada paso temporal. Online: el motor mantiene las últimas features como ventana y reentrena cada 50 ticks.",
    hyperparams: [
      "hidden_size: 10→18 unidades (según agresividad)",
      "sequence_length: 12→20 pasos de ventana",
      "learning_rate: 0.002→0.008",
      "weight_decay: 1e-5",
      "BPTT: completo sobre toda la secuencia",
      "Input size: 10 features técnicas",
      "Activación: tanh en candidato, sigmoid en puertas y salida",
    ],
    aggressiveness:
      "Conservador: red más pequeña (10 unidades) y window corta (12 pasos), LR bajo 0.002 " +
      "(más regularización, evita overfit). Temerario: 18 hidden, 20 pasos de historia, LR 0.008 " +
      "(permite capturar más patrones a riesgo de sobreajuste).",
    caveats:
      "Implementación propia en JavaScript: el backend permite escalar a secuencias más largas y estados ocultos " +
      "mayores. La versión actual usa 10-18 neuronas para mantener latencia baja en el simulador.",
  },
];

const ICONS: Record<ModelId, LucideIcon> = {
  rl: BrainCircuit,
  xgb: LineChart,
  stat: Sigma,
  rf: TreePine,
  gru: Network,
};

export function DocsPage() {
  return (
    <div className="space-y-5">
      <section className="glass-card rounded-2xl p-5">
        <div className="flex items-center gap-3">
          <div className="flex h-11 w-11 items-center justify-center rounded-full border border-emerald-500/40 bg-emerald-500/10">
            <Activity className="h-5 w-5 text-emerald-400" />
          </div>
          <div>
            <h2 className="text-sm font-bold tracking-tight">Documentación · BTC Alpha Arena</h2>
            <p className="text-[11px] text-muted-foreground">
              Cómo funciona cada modelo de IA, sus hiperparámetros y cómo el tuner controla el comportamiento
            </p>
          </div>
        </div>
      </section>

      {/* Disclaimer */}
      <div className="flex items-start gap-2 rounded-xl border border-amber-500/20 bg-amber-500/5 px-4 py-3 text-[11px] leading-relaxed text-muted-foreground">
        <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-400" />
        <p>
          Todos los modelos se entrenan y ejecutan en el navegador. Para BTC intradía (1m), los modelos
          competidores deployan distintas estrategias: aprendizaje por refuerzo online, gradient boosting,
          bayesianos estadísticos, ensemble con bagging y red recurrente. La agresividad se ajusta en caliente
          sin reentrenar todo desde cero.
        </p>
      </div>

      {/* Quick nav */}
      <section className="glass-card rounded-2xl p-5">
        <p className="mb-3 text-[10px] font-bold uppercase tracking-wider text-muted-foreground">
          Índice
        </p>
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-5">
          {MODEL_IDS.map((id) => {
            const Icon = ICONS[id];
            const meta = MODELS[id];
            return (
              <a
                key={id}
                href={`#${id}`}
                className="flex items-center gap-2 rounded-lg border border-border/40 bg-secondary/20 px-3 py-2 text-[12px] transition hover:bg-secondary/60"
              >
                <Icon className={meta.tailwindText} style={{ color: meta.colorHex }} />
                <span className="truncate">{meta.name}</span>
              </a>
            );
          })}
        </div>
      </section>

      {/* Cards de cada modelo */}
      {DOCS.map((doc) => {
        const meta = MODELS[doc.id];
        const Icon = doc.icon;
        return (
          <section
            key={doc.id}
            id={doc.id}
            className="glass-card scroll-mt-24 rounded-2xl p-5"
          >
            <div className="mb-3 flex items-center gap-3 border-b border-border/40 pb-3">
              <div
                className="flex h-10 w-10 items-center justify-center rounded-xl"
                style={{ background: `${meta.colorHex}1a` }}
              >
                <Icon className="h-5 w-5" style={{ color: meta.colorHex }} />
              </div>
              <div>
                <h3 className="text-sm font-bold tracking-tight">{doc.title}</h3>
                <p className="text-[11px] text-muted-foreground">{doc.subtitle}</p>
              </div>
            </div>

            <Section title="Cómo funciona">
              <p className="text-[12px] leading-relaxed text-muted-foreground">{doc.howItWorks}</p>
            </Section>

            <Section title="Hiperparámetros controlados por el tuner">
              <ul className="space-y-1 text-[12px] text-muted-foreground">
                {doc.hyperparams.map((h, i) => (
                  <li key={i} className="flex gap-2">
                    <span style={{ color: meta.colorHex }}>·</span>
                    <span>{h}</span>
                  </li>
                ))}
              </ul>
            </Section>

            <Section title="Cómo afecta el tuner de agresividad">
              <p className="text-[12px] leading-relaxed text-muted-foreground">{doc.aggressiveness}</p>
            </Section>

            <Section title="Caveats y limitaciones">
              <p className="text-[12px] leading-relaxed text-muted-foreground">{doc.caveats}</p>
            </Section>
          </section>
        );
      })}

      {/* Sección sobre la arena */}
      <section className="glass-card rounded-2xl p-5">
        <h3 className="mb-3 text-sm font-bold tracking-tight">Sobre la arena</h3>
        <p className="text-[12px] leading-relaxed text-muted-foreground">
          La arena ejecuta los 5 modelos en paralelo, cada uno con su propia cuenta de crédito $100,000 al 9.5% TAE.
          Comparan contra un benchmark Buy & Hold que compra BTC al inicio y mantiene el mismo interés del crédito.
          Esto permite ver si los modelos ganan <span className="font-semibold">neto de coste financiero y comisiones (0.10%)</span>.
        </p>
        <p className="mt-2 text-[12px] leading-relaxed text-muted-foreground">
          El motor arranca en backtest walk-forward sobre 320 barras históricas de 1 minuto y luego entra en modo live
          actualizando precios cada 3 segundos. Los modelos ML se reentrenan incrementalmente cada 25 ticks (75 s) para
          adaptarse a las condiciones del mercado sin reentrenar toda la historia.
        </p>
      </section>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="mb-4 border-t border-border/30 pt-3">
      <p className="mb-1.5 text-[10px] font-bold uppercase tracking-wider text-muted-foreground">
        {title}
      </p>
      {children}
    </div>
  );
}
