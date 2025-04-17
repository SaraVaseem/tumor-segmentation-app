// import Image from "next/image";
// import styles from "./page.module.css";
import { FileUpload } from "./file-upload"


export default function Home() {
  return (
        <main className="container mx-auto p-4 md:p-8">
          <div className="mx-auto max-w-2xl space-y-6">
            <div className="space-y-2 text-center">
              <h1 className="text-3xl font-bold">File Upload</h1>
              <p className="text-muted-foreground">Upload files with drag and drop or file selection</p>
            </div>
            <FileUpload
              maxFiles={5}
              maxSize={5 * 1024 * 1024} // 5MB
              accept="image/*,.pdf,.doc,.docx,.xls,.xlsx"
            />
          </div>
        </main>    
  );
}
