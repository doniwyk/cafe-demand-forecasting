import { Link, useRouterState } from "@tanstack/react-router";
import { SidebarTrigger } from "@/components/ui/sidebar";
import { Separator } from "@/components/ui/separator";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb";
import { AppTour } from "@/components/app-tour";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";

const routeTitleKeys: Record<string, string> = {
  "/": "sidebar.dashboard",
  "/forecasts": "sidebar.salesForecast",
  "/materials/daily-need": "materials.dailyMaterialRequirements",
  "/settings": "sidebar.settings",
};

export function AppHeader() {
  const routerState = useRouterState();
  const pathname = routerState.location.pathname;
  const { t, i18n } = useTranslation();
  const titleKey = routeTitleKeys[pathname] ?? "sidebar.dashboard";
  const title = t(titleKey);

  return (
    <header className="flex h-14 shrink-0 items-center gap-2 border-b px-4">
      <SidebarTrigger className="-ml-1" />
      <Separator orientation="vertical" className="mr-2 h-4! self-center" />
      <Breadcrumb>
        <BreadcrumbList>
          <BreadcrumbItem>
            <BreadcrumbLink render={<Link to="/" />}>{t("root.home")}</BreadcrumbLink>
          </BreadcrumbItem>
          {pathname !== "/" && (
            <>
              <BreadcrumbSeparator />
              <BreadcrumbItem>
                <BreadcrumbPage>{title}</BreadcrumbPage>
              </BreadcrumbItem>
            </>
          )}
        </BreadcrumbList>
      </Breadcrumb>
      <div className="ml-auto flex items-center gap-1">
        <Button
          variant="ghost"
          size="sm"
          className="text-xs font-medium text-muted-foreground"
          onClick={() => i18n.changeLanguage(i18n.language === "en" ? "id" : "en")}
        >
          {i18n.language === "en" ? "ID" : "EN"}
        </Button>
        <AppTour />
      </div>
    </header>
  );
}
