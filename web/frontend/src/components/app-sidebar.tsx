import * as React from "react"
import { PackageIcon, BarChart3Icon, LayoutDashboardIcon, SettingsIcon, ChartPie } from "lucide-react"

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
import { useTranslation } from "react-i18next"

export function AppSidebar({ ...props }: React.ComponentProps<typeof Sidebar>) {
  const { t } = useTranslation()

  const navMain = [
    {
      title: t("sidebar.dashboard"),
      url: "/",
      icon: <LayoutDashboardIcon className="size-4" />,
    },
    {
      title: t("sidebar.forecasts"),
      url: "/forecasts",
      icon: <BarChart3Icon className="size-4" />,
    },
    {
      title: t("sidebar.inventory"),
      url: "/materials/daily-need",
      icon: <PackageIcon className="size-4" />,
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
            <SidebarMenuButton size="default">
              <ChartPie className="size-4" />
              <div className="grid flex-1 text-left text-sm leading-tight">
                <span className="truncate font-semibold">HusgenCast</span>
              </div>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>
      <SidebarContent data-tour="sidebar">
        <NavMain items={navMain} />
      </SidebarContent>
      <SidebarFooter>
        <NavUser />
      </SidebarFooter>
      <SidebarRail />
    </Sidebar>
  )
}
