import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { useTranslation } from "react-i18next";
import { CHART_TOOLTIP_STYLE } from "@/lib/chart";

interface ForecastChartProps {
  selectedItem: string | null;
  forecastForItem: { date: string; predicted: number; actual: number }[];
  allItemsChartData: { date: string; predicted: number; actual: number }[];
}

export function ForecastChart({
  selectedItem,
  forecastForItem,
  allItemsChartData,
}: ForecastChartProps) {
  const { t } = useTranslation();

  return (
    <Card data-tour="forecast-chart">
      <CardHeader>
        <CardTitle>
          {selectedItem
            ? `${t("forecasts.forecast")}: ${selectedItem}`
            : t("forecasts.totalForecast")}
        </CardTitle>
      </CardHeader>
      <CardContent className="min-h-87.5">
        {selectedItem ? (
          forecastForItem.length > 0 ? (
            <ResponsiveContainer width="100%" height={310}>
              <LineChart
                data={forecastForItem}
                margin={{ top: 4, right: 4, left: 0, bottom: 0 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis
                  dataKey="date"
                  tick={{ fontSize: 11 }}
                  stroke="var(--muted-foreground)"
                />
                <YAxis tick={{ fontSize: 11 }} stroke="var(--muted-foreground)" />
                <Tooltip contentStyle={CHART_TOOLTIP_STYLE} />
                <Line
                  type="monotone"
                  dataKey="actual"
                  stroke="var(--chart-1)"
                  strokeWidth={2}
                  dot={false}
                  name={t("common.actual")}
                />
                <Line
                  type="monotone"
                  dataKey="predicted"
                  stroke="var(--chart-2)"
                  strokeWidth={2}
                  strokeDasharray="5 5"
                  dot={false}
                  name={t("forecasts.predicted")}
                />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex h-77.5 items-center justify-center text-muted-foreground">
              {t("forecasts.noForecastData")}
            </div>
          )
        ) : allItemsChartData.length > 0 ? (
          <ResponsiveContainer width="100%" height={310}>
            <LineChart data={allItemsChartData} margin={{ top: 4, right: 4, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="date" tick={{ fontSize: 11 }} stroke="var(--muted-foreground)" />
              <YAxis tick={{ fontSize: 11 }} stroke="var(--muted-foreground)" />
              <Tooltip contentStyle={CHART_TOOLTIP_STYLE} />
              <Line
                type="monotone"
                dataKey="actual"
                stroke="var(--chart-1)"
                strokeWidth={2}
                dot={false}
                name={t("common.actual")}
              />
              <Line
                type="monotone"
                dataKey="predicted"
                stroke="var(--chart-2)"
                strokeWidth={2}
                strokeDasharray="5 5"
                dot={false}
                name={t("forecasts.predicted")}
              />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <div className="flex h-77.5 items-center justify-center">
            <Skeleton className="h-full w-full" />
          </div>
        )}
      </CardContent>
    </Card>
  );
}
