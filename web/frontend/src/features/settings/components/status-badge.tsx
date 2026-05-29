import { Badge } from "@/components/ui/badge";
import { useTranslation } from "react-i18next";

export function StatusBadge({ status }: { status: string }) {
  const { t } = useTranslation();
  switch (status) {
    case "training":
      return <Badge variant="secondary">{t("settings.inProgress")}</Badge>;
    case "success":
      return <Badge variant="default">Trained</Badge>;
    case "error":
      return <Badge variant="destructive">Error</Badge>;
    default:
      return <Badge variant="outline">Not Trained</Badge>;
  }
}
