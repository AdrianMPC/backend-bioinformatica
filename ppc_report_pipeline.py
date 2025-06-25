# PPC Tracker - Flujo Completo de Vigilancia Genómica

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from typing import List
from pydantic import BaseModel
import os
import uuid
import shutil
import logging
import pandas as pd
import matplotlib.pyplot as plt
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.pairwise2 import align as pairwise_align
from collections import Counter, defaultdict
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from Bio import Entrez
import numpy as np

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ppc_tracker")

# FastAPI init
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

UPLOAD_DIR = "uploaded_files"
ALIGNMENT_DIR = "alignments"
GENE_DIR = "extracted_genes"
REFERENCE_FILE = "classical_ppc.gb"

for folder in [UPLOAD_DIR, ALIGNMENT_DIR, GENE_DIR]:
    os.makedirs(folder, exist_ok=True)

class AnalyzeRequest(BaseModel):
    file_paths: List[str]
    task_id: str
    nombre: str
    institucion: str

# EVIE AQUI CAMBIAS EL ID DEL ARCHIVO DEFAULT
def ncbi_get_reference(output_path="classical_ppc.gb", accession="AF091507.1"):
    Entrez.email = "tester@example.com"
    with Entrez.efetch(db="nucleotide", id=accession, rettype="gb", retmode="text") as handle:
        with open(output_path, "w") as f:
            f.write(handle.read())

def run_pairwise_alignment(ref_seq: str, target_seq: str) -> str:
    alignment = pairwise_align.globalxx(ref_seq, target_seq)[0]
    return alignment.seqB

def construir_matriz_variantes(reference_seq, sequences):
    mutaciones = []
    identidad = []
    dn_ds_resultados = []

    for record in sequences:
        target_seq = str(record.seq).upper()
        aligned = run_pairwise_alignment(reference_seq, target_seq)

        diffs = {}
        matches, syn, nonsyn, total_codons = 0, 0, 0, 0
        for i in range(0, min(len(reference_seq), len(aligned)) - 2, 3):
            codon_ref = reference_seq[i:i+3]
            codon_alt = aligned[i:i+3]
            if len(codon_ref) < 3 or len(codon_alt) < 3 or '-' in codon_ref or '-' in codon_alt:
                continue
            aa_ref = Seq(codon_ref).translate()
            aa_alt = Seq(codon_alt).translate()
            if codon_ref != codon_alt:
                if aa_ref == aa_alt:
                    syn += 1
                else:
                    nonsyn += 1
                diffs[i] = codon_alt
            else:
                matches += 1
            total_codons += 1

        identidad_pct = matches / total_codons * 100 if total_codons > 0 else 0
        omega = nonsyn / syn if syn > 0 else 'ND'
        identidad.append((record.id, identidad_pct))
        dn_ds_resultados.append((record.id, omega))
        mutaciones.append((record.id, diffs))

    return mutaciones, identidad, dn_ds_resultados

def analizar_pca_clustering(matriz, ids, task_id):
    X_scaled = StandardScaler().fit_transform(matriz)
    coords = PCA(n_components=2).fit_transform(X_scaled)
    labels = DBSCAN(eps=0.5, min_samples=2).fit(coords).labels_

    plt.figure(figsize=(6, 5))
    for cluster in set(labels):
        puntos = coords[labels == cluster]
        plt.scatter(puntos[:, 0], puntos[:, 1], label=f"Cluster {cluster}" if cluster != -1 else "Ruido")
    for i, txt in enumerate(ids):
        plt.annotate(txt, (coords[i, 0], coords[i, 1]), fontsize=6)
    plt.legend()
    out_path = os.path.join(ALIGNMENT_DIR, f"pca_cluster_{task_id}.png")
    plt.savefig(out_path)
    plt.close()
    return labels, out_path

def interpretar_likert(identidad: float, omega: float) -> str:
    if identidad > 99 and omega < 0.1:
        return "1 - Muy similar"
    elif identidad > 95 and omega < 0.5:
        return "2 - Leve divergencia"
    elif identidad > 90 and omega < 1.0:
        return "3 - Moderada divergencia"
    elif identidad > 85 or omega > 1.0:
        return "4 - Alta divergencia"
    else:
        return "5 - Muy divergente"

def generar_reporte(task_id, nombre, institucion, muestras, identidad, dn_ds, clusters, pca_img):
    doc_path = os.path.join(ALIGNMENT_DIR, f"reporte_vigilancia_{task_id}.pdf")
    doc = SimpleDocTemplate(doc_path, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = [
        Paragraph("Reporte Genómico - Vigilancia PPC", styles['Title']),
        Spacer(1, 12),
        Paragraph(f"Usuario: {nombre}", styles['Normal']),
        Paragraph(f"Institución: {institucion}", styles['Normal']),
        Paragraph(f"Fecha: {datetime.now().strftime('%Y-%m-%d')}", styles['Normal']),
        Spacer(1, 12),
        Paragraph("Resumen de Muestras", styles['Heading2']),
        Paragraph("\nDefiniciones: ", styles['Italic']),
        Paragraph("- Identidad%: Porcentaje de codones que coinciden con la cepa de referencia.", styles['Normal']),
        Paragraph("- dN/dS: Relación entre mutaciones no sinónimas (dN) y sinónimas (dS), indicador de presión evolutiva.", styles['Normal']),
        Paragraph("- Cluster: Agrupación basada en similitud genética por PCA+DBSCAN.", styles['Normal']),
        Spacer(1, 12)
    ]
    table_data = [["Muestra", "Identidad%", "dN/dS", "Cluster", "Escala Likert"]]
    for i, (id_, ident) in enumerate(identidad):
        omega = dict(dn_ds).get(id_, '-')
        clust = str(clusters[i]) if i < len(clusters) else '-'
        clust = "Ruido" if clust == "-1" else clust
        likert = interpretar_likert(ident, omega if isinstance(omega, float) else 0)
        table_data.append([id_, f"{ident:.2f}", f"{omega:.2f}" if isinstance(omega, float) else omega, clust, likert])

    t = Table(table_data)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
    ]))
    elements.append(t)
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("Escala de comparación tipo Likert", styles['Heading3']))
    elements.append(Paragraph("1 - Muy similar: Identidad > 99% y dN/dS < 0.1", styles['Normal']))
    elements.append(Paragraph("2 - Leve divergencia: 95–99% y dN/dS < 0.5", styles['Normal']))
    elements.append(Paragraph("3 - Moderada divergencia: 90–95% o dN/dS ~1", styles['Normal']))
    elements.append(Paragraph("4 - Alta divergencia: 85–90% o dN/dS > 1", styles['Normal']))
    elements.append(Paragraph("5 - Muy divergente: <85% o dN/dS > 2", styles['Normal']))
    elements.append(Paragraph("* Nota: dN/dS = ND significa que no se detectaron mutaciones sinónimas (dS = 0)", styles['Italic']))

    if pca_img:
        elements.append(Spacer(1, 12))
        elements.append(Paragraph("Visualización PCA/Clusters", styles['Heading2']))
        elements.append(Image(pca_img, width=400, height=300))

    elements.append(PageBreak())
    elements.append(Paragraph("Anexo: Detalles de Secuencias", styles['Heading2']))
    annex_table_data = [["ID", "Longitud", "Origen", "Fecha", "Tipo de muestra", "Autores"]]
    for record in muestras:
        features = getattr(record, 'features', [])
        fecha = origen = tipo = authors = "-"
        for f in features:
            if f.type == "source":
                origen = f.qualifiers.get("country", ["-"])[0]
                fecha = f.qualifiers.get("collection_date", ["-"])[0]
                tipo = f.qualifiers.get("isolation_source", ["-"])[0]
        authors = record.annotations.get("references", [{}])[0].authors if record.annotations.get("references") else "-"
        annex_table_data.append([record.id, str(len(record.seq)), origen, fecha, tipo, authors])
    annex_table = Table(annex_table_data, repeatRows=1)
    annex_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'TOP')
    ]))
    elements.append(annex_table)

    doc.build(elements)
    return doc_path

def run_analysis_pipeline(file_paths: List[str], task_id: str, nombre: str, institucion: str):
    if not os.path.exists(REFERENCE_FILE):
        ncbi_get_reference()

    ref_record = list(SeqIO.parse(REFERENCE_FILE, "genbank"))[0]
    reference_seq = str(ref_record.seq)
    all_seqs = []

    for path in file_paths:
        seqs = list(SeqIO.parse(path, "genbank" if path.endswith(".gb") or path.endswith(".gbk") else "fasta"))
        all_seqs.extend(seqs)

    if not all_seqs:
        logger.warning("No se encontraron secuencias válidas.")
        return

    mutaciones, identidad, dn_ds = construir_matriz_variantes(reference_seq, all_seqs)

    all_pos = sorted(set(pos for _, m in mutaciones for pos in m))
    matriz = []
    for _, mut in mutaciones:
        fila = [1 if pos in mut else 0 for pos in all_pos]
        matriz.append(fila)

    clusters, pca_img = analizar_pca_clustering(np.array(matriz), [id_ for id_, _ in identidad], task_id)
    generar_reporte(task_id, nombre, institucion, all_seqs, identidad, dn_ds, clusters, pca_img)

@app.post("/upload")
async def upload_files(nombre: str = Form(...), institucion: str = Form(...), files: List[UploadFile] = File(...)):
    task_id = str(uuid.uuid4())
    saved_files = []
    for file in files:
        path = os.path.join(UPLOAD_DIR, f"{task_id}_{file.filename}")
        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        saved_files.append(path)
    return {"task_id": task_id, "files": saved_files, "nombre": nombre, "institucion": institucion}

@app.post("/analyze")
async def analyze_files(request: AnalyzeRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(run_analysis_pipeline, request.file_paths, request.task_id, request.nombre, request.institucion)
    return {"message": "Análisis en proceso", "task_id": request.task_id}

@app.get("/report/{task_id}")
def download_report(task_id: str):
    path = os.path.join(ALIGNMENT_DIR, f"reporte_vigilancia_{task_id}.pdf")
    if os.path.exists(path):
        return FileResponse(path, media_type='application/pdf', filename=f"reporte_{task_id}.pdf")
    return {"message": "Reporte no generado."}
