import { Skeleton } from "@/components/ui/skeleton";
import { useTranslation } from "react-i18next";

export function MetricsGrid({ metrics }: { metrics: Record<string, number> | undefined }) {
  const { t } = useTranslation();

  if (!metrics) {
    return (
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-5">
        <Skeleton className="h-20" />
        <Skeleton className="h-20" />
        <Skeleton className="h-20" />
      </div>
    );
  }

  const items = [
    {
      label: t("analytics.rSquared"),
      value: metrics["r2"]?.toFixed(3) ?? "-",
      description: t("analytics.rSquaredDesc"),
    },
    {
      label: t("analytics.wmape"),
      value: `${(metrics["wmape"] ?? 0).toFixed(1)}%`,
      description: t("analytics.wmapeDesc"),
    },
    {
      label: t("analytics.mae"),
      value: metrics["mae"]?.toFixed(1) ?? "-",
      description: t("analytics.maeDesc"),
    },
    {
      label: t("analytics.rmse"),
      value: metrics["rmse"]?.toFixed(2) ?? "-",
      description: t("analytics.rmseDesc"),
    },
    {
      label: t("analytics.medianPeriodAccuracy"),
      value: `${(metrics["median_period_accuracy"] ?? 0).toFixed(1)}%`,
      description: t("analytics.medianPeriodAccuracyDesc"),
    },
    {
      label: t("analytics.periodsWithin20pct"),
      value: `${(metrics["periods_within_20pct"] ?? 0).toFixed(1)}%`,
      description: t("analytics.periodsWithin20pctDesc"),
    },
  ];

  return (
    <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
      {items.map((m) => (
        <div key={m.label} className="rounded-lg border p-4">
          <p className="text-xs text-muted-foreground">{m.label}</p>
          <p className="text-xl font-bold mt-1">{m.value}</p>
          <p className="text-xs text-muted-foreground mt-1">{m.description}</p>
        </div>
      ))}
    </div>
  );
}
