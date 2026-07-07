"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import styles from "./navbar.module.css";

export default function Navbar() {
    const pathname = usePathname();
   
    return (
      <nav className={styles.nav}>
        <span className={styles.logo}>AI Art Detector</span>
   
        <div className={styles.pill}>
          <Link
            href="/predict"
            className={`${styles.pillOption} ${pathname === "/predict" ? styles.pillActive : ""}`}
          >
            Predict
          </Link>
          <Link
            href="/history"
            className={`${styles.pillOption} ${pathname === "/history" ? styles.pillActive : ""}`}
          >
            History
          </Link>
        </div>
      </nav>
    );
  }