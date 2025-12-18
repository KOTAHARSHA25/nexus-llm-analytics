"use client";

import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { History, HelpCircle, Settings, FileText, Cpu } from "lucide-react";
import Link from "next/link";

interface SidebarProps {
  queryHistory: string[];
  onSettingsClick?: () => void;
}

export function Sidebar({ queryHistory, onSettingsClick }: SidebarProps) {
  return (
    <div className="h-full bg-card border-r p-4 space-y-6">
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm flex items-center gap-2">
            <History className="h-4 w-4" />
            Recent Queries
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          {queryHistory.map((query, index) => (
            <div key={index} className="space-y-2">
              <div className="flex items-start justify-between gap-2">
                <p className="text-xs font-medium leading-relaxed">{query}</p>
              </div>
              {index < queryHistory.length - 1 && <Separator />}
            </div>
          ))}
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm">Quick Actions</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">

          <Button variant="ghost" size="sm" className="w-full justify-start gap-2">
            <FileText className="h-4 w-4" />
            View All Reports
          </Button>
          <Button 
            variant="ghost" 
            size="sm" 
            className="w-full justify-start gap-2"
            onClick={onSettingsClick}
          >
            <Settings className="h-4 w-4" />
            Model Settings
          </Button>
          <Button variant="ghost" size="sm" className="w-full justify-start gap-2">
            <HelpCircle className="h-4 w-4" />
            Help & Support
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}
