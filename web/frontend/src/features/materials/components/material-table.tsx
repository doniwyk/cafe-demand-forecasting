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
import { DownloadIcon } from "lucide-react";
import { useTranslation } from "react-i18next";
import { DataTablePagination } from "@/components/data-table-pagination";
import { useCallback } from "react";
import { format } from "date-fns";

interface MaterialTableProps {
  isLoading: boolean;
  data: { material: string; quantity_required: number; unit: string }[];
  dateRange: { from?: Date; to?: Date };
  page: number;
  pageSize: number;
  totalPages: number;
  onPageChange: (page: number) => void;
  onPageSizeChange: (size: number) => void;
}

export function MaterialTable({
  isLoading,
  data,
  dateRange,
  page,
  pageSize,
  totalPages,
  onPageChange,
  onPageSizeChange,
}: MaterialTableProps) {
  const { t } = useTranslation();

  const exportCsv = useCallback(async () => {
    const token = localStorage.getItem("access_token");
    const params = new URLSearchParams();
    if (dateRange.from) params.set("start_date", format(dateRange.from, "yyyy-MM-dd"));
    if (dateRange.to) params.set("end_date", format(dateRange.to, "yyyy-MM-dd"));
    params.set("export", "1");

    const res = await fetch(`/api/materials/forecast?${params}`, {
      headers: { Authorization: `Bearer ${token}` },
    });
    if (!res.ok) return;
    const text = await res.text();
    const blob = new Blob([text], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `material-requirements-${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }, [dateRange]);

  return (
    <Card data-tour="daily-requirements">
      <CardHeader className="flex flex-row items-center justify-between">
        <div>
          <CardTitle>{t("materials.dailyMaterialRequirements")}</CardTitle>
          {data.length > 0 && (
            <p className="text-sm text-muted-foreground mt-1">
              {dateRange.from && format(dateRange.from, "MMM dd")} —{" "}
              {dateRange.to && format(dateRange.to, "MMM dd, yyyy")} · {data.length}{" "}
              {t("common.materials")}
            </p>
          )}
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={exportCsv}
          disabled={data.length === 0}
        >
          <DownloadIcon className="size-4" />
          {t("common.exportCsv")}
        </Button>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="space-y-2">
            {[...Array(5)].map((_, i) => (
              <Skeleton key={i} className="h-12 w-full" />
            ))}
          </div>
        ) : data.length > 0 ? (
          <>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>{t("materials.material")}</TableHead>
                  <TableHead className="text-right">{t("materials.quantityRequired")}</TableHead>
                  <TableHead className="text-right w-24">{t("common.unit")}</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {data.map((row, idx) => (
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
              onPageChange={onPageChange}
              onPageSizeChange={(s) => {
                onPageSizeChange(s);
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
  );
}
