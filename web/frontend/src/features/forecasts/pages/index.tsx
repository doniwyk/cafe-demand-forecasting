import { useState, useMemo } from "react";
import { useForecasts, useForecastSummary } from "@/features/forecasts/hooks/use-forecasts";
import { useModelType } from "@/contexts/model-context";
import { format, parseISO, addDays } from "date-fns";
import { useTranslation } from "react-i18next";
import { ItemSelector } from "@/features/forecasts/components/item-selector";
import { ForecastChart } from "@/features/forecasts/components/forecast-chart";
import { ForecastTable } from "@/features/forecasts/components/forecast-table";

export function ForecastsPage() {
  const [selectedItem, setSelectedItem] = useState<string | null>(null);
  const [dateRange, setDateRange] = useState<{ from?: Date; to?: Date }>({
    from: new Date(),
    to: addDays(new Date(), 14),
  });
  const { modelType } = useModelType();
  const { t } = useTranslation();

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
  const summary = useForecastSummary(modelType);

  const forecastForItem = useMemo(() => {
    if (!forecasts.data) return [];
    return forecasts.data.data
      .sort((a, b) => a.date.localeCompare(b.date))
      .map((f) => ({
        date: format(parseISO(f.date), "MMM dd"),
        predicted: Math.round(f.quantity_sold * 10) / 10,
      }));
  }, [forecasts.data]);

  const allItemsChartData = useMemo(() => {
    if (!forecasts.data) return [];
    const dailyTotals = new Map<string, number>();
    for (const f of forecasts.data.data) {
      dailyTotals.set(f.date, (dailyTotals.get(f.date) || 0) + f.quantity_sold);
    }
    return Array.from(dailyTotals.entries())
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([date, total]) => ({
        date: format(parseISO(date), "MMM dd"),
        total: Math.round(total),
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
