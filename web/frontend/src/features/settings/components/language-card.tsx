import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { GlobeIcon } from "lucide-react";
import { useTranslation } from "react-i18next";
import { LANGUAGES } from "@/features/settings/lib/constants";

export function LanguageCard() {
  const { t, i18n } = useTranslation();

  return (
    <Card data-tour="language">
      <CardHeader>
        <CardTitle>{t("settings.language")}</CardTitle>
        <CardDescription>{t("settings.languageDesc")}</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="flex items-center gap-3 max-w-sm">
          <GlobeIcon className="size-4 text-muted-foreground" />
          <Select
            value={i18n.language ?? undefined}
            onValueChange={(v) => i18n.changeLanguage(v ?? undefined)}
          >
            <SelectTrigger className="w-50">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {LANGUAGES.map((lang) => (
                <SelectItem key={lang.value} value={lang.value}>
                  {lang.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </CardContent>
    </Card>
  );
}
