import * as React from "react"
import { PackageIcon, BarChart3Icon, LayoutDashboardIcon, SettingsIcon } from "lucide-react"

import { NavMain } from "@/components/nav-main"
import { NavUser } from "@/components/nav-user"
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarHeader,
  SidebarRail,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar"
import { CoffeeIcon } from "lucide-react"
import { useTranslation } from "react-i18next"

const user = {
  name: "Cafe Manager",
  email: "manager@cafe.com",
  avatar: "/avatars/cafe.jpg",
}

export function AppSidebar({ ...props }: React.ComponentProps<typeof Sidebar>) {
  const { t } = useTranslation()

  const navMain = [
    {
      title: t("sidebar.dashboard"),
      url: "/",
      icon: <LayoutDashboardIcon className="size-4" />,
      isActive: true,
      items: [
        { title: t("sidebar.overview"), url: "/" },
        { title: t("sidebar.analytics"), url: "/analytics" },
      ],
    },
    {
      title: t("sidebar.forecasts"),
      url: "/forecasts",
      icon: <BarChart3Icon className="size-4" />,
      items: [
        { title: t("sidebar.salesForecast"), url: "/forecasts" },
      ],
    },
    {
      title: t("sidebar.inventory"),
      url: "/materials",
      icon: <PackageIcon className="size-4" />,
      items: [
        { title: t("materials.dailyMaterialRequirements"), url: "/materials/daily-need" },
      ],
    },
    {
      title: t("sidebar.settings"),
      url: "/settings",
      icon: <SettingsIcon className="size-4" />,
    },
  ]

  return (
    <Sidebar collapsible="icon" {...props}>
      <SidebarHeader>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton size="lg">
              <div className="flex aspect-square size-8 items-center justify-center rounded-lg bg-sidebar-primary text-sidebar-primary-foreground">
                <CoffeeIcon className="size-4" />
              </div>
              <div className="grid flex-1 text-left text-sm leading-tight">
                <span className="truncate font-semibold">{t("sidebar.brand")}</span>
                <span className="truncate text-xs">{t("sidebar.brandSub")}</span>
              </div>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>
      <SidebarContent data-tour="sidebar">
        <NavMain items={navMain} />
      </SidebarContent>
      <SidebarFooter>
        <NavUser user={user} />
      </SidebarFooter>
      <SidebarRail />
    </Sidebar>
  )
}
