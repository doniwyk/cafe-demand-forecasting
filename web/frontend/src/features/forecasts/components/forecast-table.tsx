import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { useTranslation } from "react-i18next";

interface TopSupplyItem {
  item: string;
  predicted: number;
  error_std: number;
  buffer: number;
  supply: number;
}

interface ForecastTableProps {
  items: TopSupplyItem[];
}

export function ForecastTable({ items }: ForecastTableProps) {
  const { t } = useTranslation();

  if (items.length === 0) return null;

  return (
    <Card data-tour="top-accuracy-table">
      <CardHeader>
        <CardTitle>{t("forecasts.topItemsBySupply")}</CardTitle>
      </CardHeader>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>{t("forecasts.item")}</TableHead>
              <TableHead className="text-right">{t("forecasts.predictedQty")}</TableHead>
              <TableHead className="text-right">{t("forecasts.buffer")}</TableHead>
              <TableHead className="text-right">{t("forecasts.supply")}</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {items.map((item) => (
              <TableRow key={item.item}>
                <TableCell className="font-medium">{item.item}</TableCell>
                <TableCell className="text-right">
                  {Math.round(item.predicted).toLocaleString()}
                </TableCell>
                <TableCell className="text-right">
                  {Math.round(item.buffer).toLocaleString()}
                </TableCell>
                <TableCell className="text-right font-medium">
                  {Math.round(item.supply).toLocaleString()}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
}
