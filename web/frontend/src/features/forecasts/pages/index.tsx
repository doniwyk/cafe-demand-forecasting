import { useState, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Button } from "@/components/ui/button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import {
  Command,
  CommandInput,
  CommandList,
  CommandEmpty,
  CommandGroup,
  CommandItem,
} from "@/components/ui/command";
import { Calendar } from "@/components/ui/calendar";
import { useForecasts, useForecastSummary } from "@/features/forecasts/hooks/use-forecasts";
import { useItems } from "@/features/sales/hooks/use-sales";
import { useModelType } from "@/contexts/model-context";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { format, parseISO, addDays } from "date-fns";
import { ChevronsUpDownIcon, CalendarIcon } from "lucide-react";
import { useTranslation } from "react-i18next";

export function ForecastsPage() {
  const [selectedItem, setSelectedItem] = useState<string | null>(null);
  const [open, setOpen] = useState(false);
  const [dateOpen, setDateOpen] = useState(false);
  const [dateRange, setDateRange] = useState<{ from?: Date; to?: Date }>({
    from: new Date(),
    to: addDays(new Date(), 14),
  });
  const { modelType } = useModelType();
  const { t } = useTranslation();

  const items = useItems();

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
      <Card data-tour="item-selector">
        <CardHeader>
          <CardTitle>{t("forecasts.itemSelector")}</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex gap-4 flex-wrap items-end">
            <div className="flex flex-col gap-1.5">
              <label className="text-sm font-medium">{t("forecasts.item")}</label>
              <Popover open={open} onOpenChange={setOpen}>
                <PopoverTrigger
                  render={
                    <Button
                      variant="outline"
                      role="combobox"
                      aria-expanded={open}
                      className="w-70 justify-between"
                    />
                  }
                >
                  {selectedItem || t("forecasts.selectItem")}
                  <ChevronsUpDownIcon className="ml-auto size-4 shrink-0 opacity-50" />
                </PopoverTrigger>
                <PopoverContent className="w-70 p-0" align="start">
                  <Command>
                    <CommandInput placeholder={t("forecasts.searchItems")} />
                    <CommandList>
                      <CommandEmpty>{t("forecasts.noItemsFound")}</CommandEmpty>
                      <CommandGroup>
                        {items.data?.map((item) => (
                          <CommandItem
                            key={item.name}
                            value={item.name}
                            onSelect={(value) => {
                              setSelectedItem(value === selectedItem ? null : value);
                              setOpen(false);
                            }}
                          >
                            {item.name}
                          </CommandItem>
                        ))}
                      </CommandGroup>
                    </CommandList>
                  </Command>
                </PopoverContent>
              </Popover>
            </div>

            <div className="flex flex-col gap-1.5">
              <label className="text-sm font-medium">{t("forecasts.dateRange")}</label>
              <Popover open={dateOpen} onOpenChange={setDateOpen}>
                <PopoverTrigger
                  render={
                    <Button variant="outline" className="w-70 justify-start text-left font-normal">
                      <CalendarIcon className="mr-2 size-4 shrink-0" />
                      {dateRange.from ? (
                        dateRange.to ? (
                          <>
                            {format(dateRange.from, "MMM dd, yyyy")} —{" "}
                            {format(dateRange.to, "MMM dd, yyyy")}
                          </>
                        ) : (
                          format(dateRange.from, "MMM dd, yyyy")
                        )
                      ) : (
                        <span className="text-muted-foreground">
                          {t("forecasts.selectDateRange")}
                        </span>
                      )}
                    </Button>
                  }
                />
                <PopoverContent className="w-auto p-0" align="start">
                  <Calendar
                    mode="range"
                    selected={dateRange as { from: Date; to?: Date }}
                    onSelect={(range) => setDateRange(range ?? {})}
                    numberOfMonths={2}
                  />
                </PopoverContent>
              </Popover>
            </div>

            {(selectedItem || dateRange.from || dateRange.to) && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => {
                  setSelectedItem(null);
                  setDateRange({});
                }}
              >
                {t("common.clear")}
              </Button>
            )}
          </div>
          {selectedItem && (
            <div className="mt-3 flex items-center gap-2">
              <Badge variant="secondary">{selectedItem}</Badge>
            </div>
          )}
        </CardContent>
      </Card>

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
              <div className="flex flex-col gap-4">
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
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "var(--popover)",
                        border: "1px solid var(--border)",
                        borderRadius: "var(--radius)",
                        fontSize: 12,
                      }}
                    />
                    <Line
                      type="monotone"
                      dataKey="predicted"
                      stroke="var(--chart-2)"
                      strokeWidth={2}
                      dot={false}
                      name={t("forecasts.predicted")}
                    />
                  </LineChart>
                </ResponsiveContainer>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>{t("forecasts.date")}</TableHead>
                      <TableHead className="text-right">{t("forecasts.predictedQty")}</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {(() => {
                      const data =
                        forecasts.data?.data
                          .filter((f) => f.item === selectedItem)
                          .sort((a, b) => a.date.localeCompare(b.date)) ?? [];
                      return data.map((f, idx) => (
                        <TableRow key={`${f.date}-${idx}`}>
                          <TableCell className="font-medium">
                            {format(parseISO(f.date), "MMM dd, yyyy")}
                          </TableCell>
                          <TableCell className="text-right">
                            {Math.round(f.quantity_sold * 10) / 10}
                          </TableCell>
                        </TableRow>
                      ));
                    })()}
                  </TableBody>
                </Table>
              </div>
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
                <Tooltip
                  contentStyle={{
                    backgroundColor: "var(--popover)",
                    border: "1px solid var(--border)",
                    borderRadius: "var(--radius)",
                    fontSize: 12,
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="total"
                  stroke="var(--chart-1)"
                  strokeWidth={2}
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

      {topForecastItems.length > 0 && (
        <Card data-tour="top-accuracy-table">
          <CardHeader>
            <CardTitle>{t("forecasts.topItemsByAccuracy")}</CardTitle>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Item</TableHead>
                  <TableHead className="text-right">{t("forecasts.actualQty")}</TableHead>
                  <TableHead className="text-right">{t("forecasts.predictedQty")}</TableHead>
                  <TableHead className="text-right">{t("forecasts.accuracy")}</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {topForecastItems.map((item) => (
                  <TableRow key={item.item}>
                    <TableCell className="font-medium">{item.item}</TableCell>
                    <TableCell className="text-right">
                      {Math.round(item.quantity_sold).toLocaleString()}
                    </TableCell>
                    <TableCell className="text-right">
                      {Math.round(item.predicted).toLocaleString()}
                    </TableCell>
                    <TableCell className="text-right">
                      <Badge
                        variant={
                          item.accuracy_pct >= 70
                            ? "default"
                            : item.accuracy_pct >= 50
                              ? "secondary"
                              : "destructive"
                        }
                      >
                        {item.accuracy_pct.toFixed(1)}%
                      </Badge>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
