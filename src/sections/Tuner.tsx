import { Gauge, RotateCcw } from "lucide-react";
import { MODELS, AGGRESSION_LABELS, AGGRESSION_SHORT, type ModelId, type AggressionLevel } from "@/lib/registry";
import { cn } from "@/lib/utils";

interface Props {
  modelId: ModelId;
  current: AggressionLevel;
  onChange: (level: AggressionLevel) => void;
  scaleLabel: string;
}

export function Tuner({ modelId, current, onChange, scaleLabel }: Props) {
  const meta = MODELS[modelId];
  const levels: AggressionLevel[] = [0, 1, 2, 3, 4];

  return (
    <section className="glass-card rounded-2xl p-5">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <Gauge className={`h-4 w-4 ${meta.tailwindText}`} />
          <div>
            <h3 className="text-sm font-bold tracking-tight">Tuner online · agresividad</h3>
            <p className="text-[11px] text-muted-foreground">
              Ajuste inmediato de hiperparámetros · Restaurar neutral reinicia el perfil
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span className={`rounded-full border border-border/60 bg-secondary/50 px-3 py-1 text-[11px] font-bold uppercase tracking-wider ${meta.tailwindText}`}>
            {scaleLabel}
          </span>
          <button
            type="button"
            onClick={() => onChange(2)}
            className="flex items-center gap-1 rounded-md border border-border/60 bg-secondary/40 px-2 py-1 text-[10px] text-muted-foreground transition hover:bg-secondary/80 hover:text-foreground"
          >
            <RotateCcw className="h-3 w-3" />
            Neutral
          </button>
        </div>
      </div>

      {/* Pista de niveles */}
      <div className="flex items-center gap-2">
        {levels.map((lvl) => {
          const active = lvl === current;
          const filled = lvl <= current;
          return (
            <button
              key={lvl}
              type="button"
              onClick={() => onChange(lvl)}
              className={cn(
                "flex-1 rounded-xl border px-3 py-3 text-center transition-all",
                active
                  ? `${meta.tailwindRing} border-transparent shadow-lg`
                  : "border-border/50 bg-secondary/30 hover:bg-secondary/60"
              )}
              style={
                active
                  ? { background: `${meta.colorHex}1a`, borderColor: `${meta.colorHex}66` }
                  : filled
                    ? { background: `${meta.colorHex}0d` }
                    : undefined
              }
            >
              <p
                className={`text-[11px] font-bold uppercase tracking-wider ${
                  active ? meta.tailwindText : "text-muted-foreground"
                }`}
              >
                {AGGRESSION_SHORT[lvl]}
              </p>
              <p className="mt-0.5 text-[9px] leading-tight text-muted-foreground">
                {AGGRESSION_LABELS[lvl]}
              </p>
              {/* Barra de intensidad */}
              <div className="mt-2 flex h-1 gap-px">
                {levels.map((dot) => (
                  <span
                    key={dot}
                    className="flex-1 rounded-full"
                    style={{
                      background: dot <= lvl ? meta.colorHex : "hsl(220 22% 22%)",
                      opacity: dot <= lvl ? (active ? 1 : 0.5) : 1,
                    }}
                  />
                ))}
              </div>
            </button>
          );
        })}
      </div>

      <p className="mt-3 text-center text-[10px] text-muted-foreground">
        Conservador = más prudente (umbrales altos, poca exploración) · Temerario = más operativa (umbrales bajos, alta exploración)
      </p>
    </section>
  );
}
