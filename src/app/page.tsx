// import Image from "next/image";
// import styles from "./page.module.css";
import { FileUpload } from "./file-upload"


export default function Home() {
  return (
        <main className="container mx-auto p-4 md:p-8">
          <div className="mx-auto max-w-2xl space-y-6">
            <div className="space-y-2 text-center">
            <FileUpload/>
            </div>
          </div>
        </main>    
  );
}
