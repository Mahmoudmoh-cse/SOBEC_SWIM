import clsx from "clsx";

type StatusPillProps = {
  children: React.ReactNode;
  tone?: "water" | "mint" | "coral" | "violet" | "slate";
};

const toneClasses = {
  water: "bg-blue-50 text-water",
  mint: "bg-emerald-50 text-mint",
  coral: "bg-orange-50 text-coral",
  violet: "bg-violet-50 text-violet",
  slate: "bg-slate-100 text-slate-700"
};

export function StatusPill({ children, tone = "slate" }: StatusPillProps) {
  return (
    <span className={clsx("inline-flex items-center rounded px-2 py-1 text-xs font-semibold", toneClasses[tone])}>
      {children}
    </span>
  );
}
