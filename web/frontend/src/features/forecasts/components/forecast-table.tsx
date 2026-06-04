import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { useTranslation } from "react-i18next";

interface ForecastTableProps {
  items: { item: string; quantity_sold: number; predicted: number; accuracy_pct: number }[];
}

export function ForecastTable({ items }: ForecastTableProps) {
  const { t } = useTranslation();

  if (items.length === 0) return null;

  return (
    <Card data-tour="top-accuracy-table">
      <CardHeader>
        <CardTitle>{t("forecasts.topItemsByAccuracy")}</CardTitle>
      </CardHeader>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>{t("forecasts.item")}</TableHead>
              <TableHead className="text-right">{t("forecasts.actualQty")}</TableHead>
              <TableHead className="text-right">{t("forecasts.predictedQty")}</TableHead>
              <TableHead className="text-right">{t("forecasts.accuracy")}</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {items.map((item) => (
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
  );
}
