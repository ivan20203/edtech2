"use client";

import { Button } from "./ui/button";
import { GitIcon, VercelIcon } from "./icons";
import Link from "next/link";

export const Navbar = () => {
  return (
    <div className="p-2 flex flex-row gap-2 justify-between">
      <div className="flex gap-2">
        <Link href="/">
          <Button variant="ghost" size="sm">
            Home
          </Button>
        </Link>
        <Link href="/podagent-demo">
          <Button variant="ghost" size="sm">
            PodAgent Demo
          </Button>
        </Link>
      </div>
    </div>
  );
};
