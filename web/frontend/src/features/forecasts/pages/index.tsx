import { useState, useMemo, useEffect } from "react";
import { useForecasts, useForecastSummary } from "@/features/forecasts/hooks/use-forecasts";
import { useModelType } from "@/contexts/model-context";
import { format, parseISO, addDays } from "date-fns";
import { useTranslation } from "react-i18next";
import { ItemSelector } from "@/features/forecasts/components/item-selector";
import { ForecastChart } from "@/features/forecasts/components/forecast-chart";
import { ForecastTable } from "@/features/forecasts/components/forecast-table";

export function ForecastsPage() {
  const [selectedItem, setSelectedItem] = useState<string | null>(null);
  const [dateRange, setDateRange] = useState<{ from?: Date; to?: Date }>({});
  const { modelType } = useModelType();
  const { t } = useTranslation();

  const summary = useForecastSummary(modelType);

  // Default date range starts from training cutoff + 30 days of predictions
  useEffect(() => {
    if (summary.data?.latest_training_date && !dateRange.from) {
      const cutoff = parseISO(summary.data.latest_training_date);
      setDateRange({
        from: cutoff,
        to: addDays(cutoff, 30),
      });
    }
  }, [summary.data?.latest_training_date]);

  const forecastParams = useMemo(() => {
    const p: { item?: string; start_date: string; end_date: string; model_type?: string } = {
      model_type: modelType,
      start_date: format(dateRange.from ?? new Date(), "yyyy-MM-dd"),
      end_date: format(dateRange.to ?? addDays(new Date(), 14), "yyyy-MM-dd"),
    };
    if (selectedItem) p.item = selectedItem;
    return p;
  }, [selectedItem, dateRange, modelType]);

  const forecasts = useForecasts(forecastParams);

  const forecastForItem = useMemo(() => {
    if (!forecasts.data) return [];
    return forecasts.data.data
      .sort((a, b) => a.date.localeCompare(b.date))
      .map((f) => ({
        date: format(parseISO(f.date), "MMM dd"),
        predicted: Math.round(f.quantity_sold * 10) / 10,
        actual: Math.round(f.actual * 10) / 10,
      }));
  }, [forecasts.data]);

  const allItemsChartData = useMemo(() => {
    if (!forecasts.data) return [];
    const dailyTotals = new Map<string, { predicted: number; actual: number }>();
    for (const f of forecasts.data.data) {
      const existing = dailyTotals.get(f.date) || { predicted: 0, actual: 0 };
      dailyTotals.set(f.date, {
        predicted: existing.predicted + f.quantity_sold,
        actual: existing.actual + f.actual,
      });
    }
    return Array.from(dailyTotals.entries())
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([date, totals]) => ({
        date: format(parseISO(date), "MMM dd"),
        predicted: Math.round(totals.predicted),
        actual: Math.round(totals.actual),
      }));
  }, [forecasts.data]);

  const topForecastItems = useMemo(() => {
    if (!summary.data) return [];
    return summary.data.top_items.slice(0, 10);
  }, [summary.data]);

  return (
    <div className="flex flex-1 flex-col gap-6 p-4">
      <ItemSelector
        selectedItem={selectedItem}
        onSelectItem={setSelectedItem}
        dateRange={dateRange}
        onDateRangeChange={setDateRange}
      />

      <ForecastChart
        selectedItem={selectedItem}
        forecastForItem={forecastForItem}
        allItemsChartData={allItemsChartData}
      />

      <ForecastTable items={topForecastItems} />
    </div>
  );
}
