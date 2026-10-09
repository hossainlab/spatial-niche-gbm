# Export SPATA (v1.x) objects from the Ravi et al. 2022 / NePSTA cohort
# (University Clinic Freiburg, "UKF") to plain files, without needing SPATA2.
#
# For each <sample>.RDS this writes <out>/<sample>/:
#   counts.mtx   raw UMI counts (genes x spots, MatrixMarket)
#   genes.txt    gene symbols (rows of counts.mtx)
#   barcodes.txt spot barcodes (columns of counts.mtx)
#   spots.csv    barcode, x, y and every per-spot feature in the object
#                (including `segmentation`: the neuropathologist's manual
#                H&E-based region annotation, imported into SPATA per
#                Ritter et al. 2025, Nat Cancer, Methods)
#
# Usage: Rscript scripts/18_export_spata_objects.R <in_dir> <out_dir>

suppressPackageStartupMessages(library(Matrix))
args <- commandArgs(trailingOnly = TRUE)
in_dir <- args[1]; out_dir <- args[2]
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

for (f in list.files(in_dir, pattern = "\\.RDS$", full.names = TRUE)) {
  o <- tryCatch(suppressWarnings(readRDS(f)), error = function(e) NULL)
  if (is.null(o)) { message("SKIP (unreadable): ", f); next }
  a <- attributes(o)
  for (s in names(a$data)) {
    od <- file.path(out_dir, s)
    if (file.exists(file.path(od, "spots.csv"))) next
    dir.create(od, showWarnings = FALSE)
    counts <- a$data[[s]]$counts
    writeMM(counts, file.path(od, "counts.mtx"))
    writeLines(rownames(counts), file.path(od, "genes.txt"))
    writeLines(colnames(counts), file.path(od, "barcodes.txt"))
    coords <- a$coordinates[[s]][, c("barcodes", "x", "y")]
    fd <- a$fdata[[s]]
    spots <- merge(coords, fd, by = "barcodes", all.x = TRUE)
    write.csv(spots, file.path(od, "spots.csv"), row.names = FALSE)
    seg <- if ("segmentation" %in% names(spots)) table(spots$segmentation) else "none"
    message(s, ": ", ncol(counts), " spots; segmentation: ",
            paste(names(seg), seg, sep = "=", collapse = ", "))
  }
}
