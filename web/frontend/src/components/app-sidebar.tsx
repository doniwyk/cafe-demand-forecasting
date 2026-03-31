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

const data = {
  user: {
    name: "Cafe Manager",
    email: "manager@cafe.com",
    avatar: "/avatars/cafe.jpg",
  },
  navMain: [
    {
      title: "Dashboard",
      url: "/",
      icon: <LayoutDashboardIcon className="size-4" />,
      isActive: true,
      items: [
        { title: "Overview", url: "/" },
        { title: "Analytics", url: "/analytics" },
      ],
    },
    {
      title: "Forecasts",
      url: "/forecasts",
      icon: <BarChart3Icon className="size-4" />,
      items: [
        { title: "Sales Forecast", url: "/forecasts" },
      ],
    },
    {
      title: "Inventory",
      url: "/materials",
      icon: <PackageIcon className="size-4" />,
      items: [
        { title: "Raw Materials", url: "/materials" },
      ],
    },
    {
      title: "Settings",
      url: "/settings",
      icon: <SettingsIcon className="size-4" />,
      items: [
        { title: "General", url: "/settings" },
      ],
    },
  ],
}

export function AppSidebar({ ...props }: React.ComponentProps<typeof Sidebar>) {
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
                <span className="truncate font-semibold">Cafe Supply</span>
                <span className="truncate text-xs">Forecasting</span>
              </div>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>
      <SidebarContent>
        <NavMain items={data.navMain} />
      </SidebarContent>
      <SidebarFooter>
        <NavUser user={data.user} />
      </SidebarFooter>
      <SidebarRail />
    </Sidebar>
  )
}
