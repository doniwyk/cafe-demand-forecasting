import { useState, useMemo, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
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
import { useDailyNeed } from "@/features/materials/hooks/use-materials";
import { format, addDays } from "date-fns";
import { Calendar } from "@/components/ui/calendar";
import { ChevronsUpDownIcon, DownloadIcon, CalendarIcon } from "lucide-react";
import { useTranslation } from "react-i18next";
import { DataTablePagination } from "@/components/data-table-pagination";

export function DailyNeedPage() {
  const [selectedMaterial, setSelectedMaterial] = useState<string | null>(null);
  const [dateRange, setDateRange] = useState<{ from?: Date; to?: Date }>({
    from: new Date(),
    to: addDays(new Date(), 14),
  });
  const [materialOpen, setMaterialOpen] = useState(false);
  const [dateOpen, setDateOpen] = useState(false);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);
  const { t } = useTranslation();

  const dateParams = useMemo(() => {
    const p: {
      start_date: string;
      end_date: string;
      page: number;
      page_size: number;
      material?: string;
    } = {
      start_date: format(dateRange.from ?? new Date(), "yyyy-MM-dd"),
      end_date: format(dateRange.to ?? addDays(new Date(), 14), "yyyy-MM-dd"),
      page,
      page_size: pageSize,
    };
    if (selectedMaterial) p.material = selectedMaterial;
    return p;
  }, [dateRange.from, dateRange.to, page, pageSize, selectedMaterial]);

  const dailyNeed = useDailyNeed(dateParams);

  const materials = useMemo(() => {
    if (!dailyNeed.data) return [];
    return [...new Set(dailyNeed.data.data.map((d) => d.raw_material))].sort();
  }, [dailyNeed.data]);

  const aggregatedData = useMemo(() => {
    if (!dailyNeed.data) return [];
    return dailyNeed.data.data.map((d) => ({
      material: d.raw_material,
      quantity_required: d.quantity_required,
      unit: d.unit ?? "",
    }));
  }, [dailyNeed.data]);

  const totalPages = dailyNeed.data ? Math.ceil(dailyNeed.data.total / pageSize) : 0;

  const exportCsv = useCallback(() => {
    const headers = ["Material", "Total Quantity Required"];
    const rows = aggregatedData.map((d) => [d.material, d.quantity_required]);
    const csv = [headers, ...rows].map((r) => r.join(",")).join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `material-requirements-${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }, [aggregatedData]);

  return (
    <div className="flex flex-1 flex-col gap-6 p-4">
      <Card data-tour="material-filter">
        <CardHeader>
          <CardTitle>{t("materials.materialFilter")}</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex gap-4 flex-wrap items-end">
            <div className="flex flex-col gap-1.5">
              <label className="text-sm font-medium">{t("materials.dateRange")}</label>
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
                          {t("materials.selectDateRange")}
                        </span>
                      )}
                    </Button>
                  }
                />
                <PopoverContent className="w-auto p-0" align="start">
                  <Calendar
                    mode="range"
                    selected={dateRange as { from: Date; to?: Date }}
                    onSelect={(range) =>
                      setDateRange(range ?? { from: new Date(), to: addDays(new Date(), 7) })
                    }
                    numberOfMonths={2}
                  />
                </PopoverContent>
              </Popover>
            </div>
            <div className="flex flex-col gap-1.5">
              <label className="text-sm font-medium">{t("materials.material")}</label>
              <Popover open={materialOpen} onOpenChange={setMaterialOpen}>
                <PopoverTrigger
                  render={
                    <Button
                      variant="outline"
                      role="combobox"
                      aria-expanded={materialOpen}
                      className="w-70 justify-between"
                    />
                  }
                >
                  {selectedMaterial || t("materials.selectMaterial")}
                  <ChevronsUpDownIcon className="ml-auto size-4 shrink-0 opacity-50" />
                </PopoverTrigger>
                <PopoverContent className="w-70 p-0" align="start">
                  <Command>
                    <CommandInput placeholder={t("materials.searchMaterials")} />
                    <CommandList>
                      <CommandEmpty>{t("materials.noMaterialsFound")}</CommandEmpty>
                      <CommandGroup>
                        {materials.map((mat) => (
                          <CommandItem
                            key={mat}
                            value={mat}
                            onSelect={(value) => {
                              setSelectedMaterial(value === selectedMaterial ? null : value);
                              setMaterialOpen(false);
                            }}
                          >
                            {mat}
                          </CommandItem>
                        ))}
                      </CommandGroup>
                    </CommandList>
                  </Command>
                </PopoverContent>
              </Popover>
            </div>
            {(selectedMaterial || dateRange.from || dateRange.to) && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => {
                  setSelectedMaterial(null);
                  setDateRange({ from: new Date(), to: addDays(new Date(), 7) });
                }}
              >
                {t("common.clear")}
              </Button>
            )}
          </div>
        </CardContent>
      </Card>

      <Card data-tour="daily-requirements">
        <CardHeader className="flex flex-row items-center justify-between">
          <div>
            <CardTitle>{t("materials.dailyMaterialRequirements")}</CardTitle>
            {aggregatedData.length > 0 && (
              <p className="text-sm text-muted-foreground mt-1">
                {dateRange.from && format(dateRange.from, "MMM dd")} —{" "}
                {dateRange.to && format(dateRange.to, "MMM dd, yyyy")} · {aggregatedData.length}{" "}
                materials
              </p>
            )}
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={exportCsv}
            disabled={aggregatedData.length === 0}
          >
            <DownloadIcon className="size-4" />
            Export CSV
          </Button>
        </CardHeader>
        <CardContent>
          {dailyNeed.isLoading ? (
            <div className="space-y-2">
              {[...Array(5)].map((_, i) => (
                <Skeleton key={i} className="h-12 w-full" />
              ))}
            </div>
          ) : aggregatedData.length > 0 ? (
            <>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>{t("materials.material")}</TableHead>
                    <TableHead className="text-right">{t("materials.quantityRequired")}</TableHead>
                    <TableHead className="text-right w-24">Unit</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {aggregatedData.map((row, idx) => (
                    <TableRow key={`${row.material}-${idx}`}>
                      <TableCell className="font-medium">{row.material}</TableCell>
                      <TableCell className="text-right tabular-nums">
                        {row.quantity_required.toLocaleString()}
                      </TableCell>
                      <TableCell className="text-right text-muted-foreground">{row.unit}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
              <DataTablePagination
                page={page}
                pageSize={pageSize}
                totalPages={totalPages}
                onPageChange={setPage}
                onPageSizeChange={(s) => {
                  setPageSize(s);
                  setPage(1);
                }}
              />
            </>
          ) : (
            <div className="text-center py-8 text-muted-foreground">
              {t("materials.noMaterialData")}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
