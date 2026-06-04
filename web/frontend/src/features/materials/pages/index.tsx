import { useState, useMemo } from "react";
import { useDailyNeed } from "@/features/materials/hooks/use-materials";
import { format, addDays } from "date-fns";
import { useTranslation } from "react-i18next";
import { MaterialFilter } from "@/features/materials/components/material-filter";
import { MaterialTable } from "@/features/materials/components/material-table";

export function DailyNeedPage() {
  const [selectedMaterial, setSelectedMaterial] = useState<string | null>(null);
  const [dateRange, setDateRange] = useState<{ from?: Date; to?: Date }>({
    from: new Date(),
    to: addDays(new Date(), 14),
  });
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

  return (
    <div className="flex flex-1 flex-col gap-6 p-4">
      <MaterialFilter
        materials={materials}
        selectedMaterial={selectedMaterial}
        onSelectMaterial={setSelectedMaterial}
        dateRange={dateRange}
        onDateRangeChange={setDateRange}
      />

      <MaterialTable
        isLoading={dailyNeed.isLoading}
        data={aggregatedData}
        dateRange={dateRange}
        page={page}
        pageSize={pageSize}
        totalPages={totalPages}
        onPageChange={setPage}
        onPageSizeChange={(s) => {
          setPageSize(s);
          setPage(1);
        }}
      />
    </div>
  );
}
