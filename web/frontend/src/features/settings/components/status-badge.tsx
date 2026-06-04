import { Badge } from "@/components/ui/badge";
import { useTranslation } from "react-i18next";
import type { ModelStatus } from "@/features/settings/types";

export function StatusBadge({ status }: { status: ModelStatus }) {
  const { t } = useTranslation();
  switch (status) {
    case "training":
      return <Badge variant="secondary">{t("settings.inProgress")}</Badge>;
    case "success":
      return <Badge variant="default">{t("settings.trained")}</Badge>;
    case "error":
      return <Badge variant="destructive">{t("settings.error")}</Badge>;
    default:
      return <Badge variant="outline">{t("settings.notTrained")}</Badge>;
  }
}
