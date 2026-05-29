import type { LucideIcon } from "lucide-react";

type MetricCardProps = {
  label: string;
  value: string | number;
  detail: string;
  icon: LucideIcon;
  tone?: "water" | "mint" | "coral" | "violet";
};

const tones = {
  water: "bg-blue-50 text-water",
  mint: "bg-emerald-50 text-mint",
  coral: "bg-orange-50 text-coral",
  violet: "bg-violet-50 text-violet"
};

export function MetricCard({ label, value, detail, icon: Icon, tone = "water" }: MetricCardProps) {
  return (
    <section className="panel p-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="label">{label}</p>
          <p className="mt-2 text-3xl font-bold text-ink">{value}</p>
        </div>
        <span className={`flex h-10 w-10 items-center justify-center rounded-md ${tones[tone]}`}>
          <Icon size={20} />
        </span>
      </div>
      <p className="mt-3 text-sm text-slate-500">{detail}</p>
    </section>
  );
}
