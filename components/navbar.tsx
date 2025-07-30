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
        <Link href="/demo">
          <Button variant="ghost" size="sm">
            Demos
          </Button>
        </Link>
        <Link href="/library">
          <Button variant="ghost" size="sm">
            Library
          </Button>
        </Link>
      </div>
    </div>
  );
};
