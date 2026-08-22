// frontend/app/layout.tsx
import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';

const inter = Inter({ subsets: ['latin'], variable: '--font-inter' });

export const metadata: Metadata = {
  title: 'Smart Traffic AI — Intelligence & Violation Detection',
  description:
    'Production-grade traffic camera analysis system. Detects vehicles, tracks movement, and identifies traffic violations using YOLOv8 + OpenCV.',
  keywords: ['traffic AI', 'violation detection', 'computer vision', 'YOLO', 'object detection'],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={inter.variable}>
      <body className={inter.className}>{children}</body>
    </html>
  );
}
