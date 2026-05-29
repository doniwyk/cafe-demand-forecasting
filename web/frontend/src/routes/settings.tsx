import { createFileRoute } from "@tanstack/react-router";
import { SettingsPage } from "@/features/settings/pages";

export const Route = createFileRoute("/settings")({
  component: SettingsPage,
});
