from __future__ import annotations

from dataclasses import dataclass


GENERIC_MARKERS = {
    "ACTB",
    "B2M",
    "EEF1A1",
    "FTL",
    "FTH1",
    "GAPDH",
    "HSP90AA1",
    "HSP90AB1",
    "MALAT1",
    "NEAT1",
    "RPLP1",
    "TMSB10",
    "TMSB4X",
    "TPT1",
    "UBC",
    "VIM",
}


@dataclass(frozen=True)
class MarkerProfile:
    canonical_label: str
    positive_markers: tuple[str, ...]
    negative_markers: tuple[str, ...] = ()
    confusion_labels: tuple[str, ...] = ()
    ontology_ids: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()


MARKER_PROFILES = {
    "leukocyte": MarkerProfile(
        canonical_label="leukocyte",
        positive_markers=("PTPRC", "B2M", "LST1", "TYROBP", "FCER1G"),
        negative_markers=("EPCAM", "PECAM1", "COL1A1", "HBB"),
        confusion_labels=("myeloid leukocyte", "t cell", "b cell"),
        aliases=("white blood cell",),
    ),
    "myeloid leukocyte": MarkerProfile(
        canonical_label="myeloid leukocyte",
        positive_markers=("LYZ", "FCER1G", "TYROBP", "CTSS", "AIF1"),
        negative_markers=("CD3D", "MS4A1", "EPCAM", "COL1A1"),
        confusion_labels=("monocyte", "macrophage", "dendritic cell", "granulocyte"),
        aliases=("myeloid cell",),
    ),
    "granulocyte": MarkerProfile(
        canonical_label="granulocyte",
        positive_markers=("S100A8", "S100A9", "CSF3R", "FPR1", "FPR2"),
        negative_markers=("CD3D", "MS4A1", "PECAM1", "COL1A1"),
        confusion_labels=("neutrophil", "basophil", "mast cell", "monocyte"),
        aliases=(),
    ),
    "mononuclear phagocyte": MarkerProfile(
        canonical_label="mononuclear phagocyte",
        positive_markers=("LST1", "FCER1G", "TYROBP", "CTSS", "AIF1"),
        negative_markers=("CD3D", "MS4A1", "PECAM1", "EPCAM"),
        confusion_labels=("monocyte", "macrophage", "dendritic cell"),
        aliases=("mp",),
    ),
    "monocyte": MarkerProfile(
        canonical_label="monocyte",
        positive_markers=("S100A8", "S100A9", "FCN1", "CTSS", "LYZ", "SAT1", "VCAN", "IL1B", "CXCL8", "PLAUR"),
        negative_markers=("C1QA", "C1QB", "C1QC", "CLEC4C", "CLEC9A", "MS4A1"),
        confusion_labels=("non-classical monocyte", "macrophage", "neutrophil", "dendritic cell", "monocyte-derived dendritic cell"),
        ontology_ids=("CL:0000576",),
        aliases=("classical monocyte", "intermediate monocyte"),
    ),
    "non-classical monocyte": MarkerProfile(
        canonical_label="non-classical monocyte",
        positive_markers=("FCGR3A", "CX3CR1", "IFITM3", "LST1", "SAT1", "TYMP"),
        negative_markers=("S100A8", "S100A9", "CLEC4C", "CLEC9A", "MS4A1"),
        confusion_labels=("monocyte", "macrophage", "mononuclear phagocyte"),
        aliases=("cd16+ non-classical monocyte",),
    ),
    "macrophage": MarkerProfile(
        canonical_label="macrophage",
        positive_markers=("APOE", "C1QA", "C1QB", "C1QC", "CD163", "FCGR3A", "MS4A6A", "CTSB", "CTSD", "GPNMB"),
        negative_markers=("S100A8", "S100A9", "CLEC4C", "CLEC9A", "MS4A1"),
        confusion_labels=("monocyte", "dendritic cell", "microglial cell"),
        ontology_ids=("CL:0000235",),
        aliases=("tissue-resident macrophage",),
    ),
    "microglial cell": MarkerProfile(
        canonical_label="microglial cell",
        positive_markers=("TMEM119", "P2RY12", "CX3CR1", "HEXB", "SALL1"),
        negative_markers=("S100A8", "S100A9", "MS4A1", "CD3D", "EPCAM"),
        confusion_labels=("macrophage", "glial cell", "mononuclear phagocyte"),
        aliases=("microglia",),
    ),
    "neutrophil": MarkerProfile(
        canonical_label="neutrophil",
        positive_markers=("S100A8", "S100A9", "CSF3R", "CXCL8", "FCGR3B", "FPR1", "FPR2", "IL1R2", "AQP9", "S100A12"),
        negative_markers=("FCN1", "APOE", "C1QA", "KIT", "TPSAB1"),
        confusion_labels=("granulocyte", "monocyte", "macrophage", "mast cell", "basophil"),
        ontology_ids=("CL:0000775",),
    ),
    "basophil": MarkerProfile(
        canonical_label="basophil",
        positive_markers=("ENPP3", "HDC", "CCR3", "IL3RA", "CLC", "GATA2"),
        negative_markers=("TPSAB1", "TPSB2", "KIT", "FCGR3B", "S100A8"),
        confusion_labels=("mast cell", "granulocyte", "neutrophil"),
        ontology_ids=("CL:0000767",),
        aliases=("basophilic granulocyte",),
    ),
    "mast cell": MarkerProfile(
        canonical_label="mast cell",
        positive_markers=("KIT", "TPSAB1", "TPSB2", "CPA3", "MS4A2", "HPGDS", "HDC", "RGS13", "GATA2", "SIGLEC6"),
        negative_markers=("ENPP3", "FCGR3B", "MS4A1", "JCHAIN"),
        confusion_labels=("basophil", "neutrophil", "plasma cell"),
        ontology_ids=("CL:0000097",),
    ),
    "plasmacytoid dendritic cell": MarkerProfile(
        canonical_label="plasmacytoid dendritic cell",
        positive_markers=("IL3RA", "CLEC4C", "NRP1", "TCF4", "GZMB", "LILRB4", "LILRA4", "IRF7", "BTLA"),
        negative_markers=("CD1C", "CLEC10A", "CLEC9A", "S100A8", "S100A9"),
        confusion_labels=("dendritic cell", "conventional dendritic cell 2", "monocyte-derived dendritic cell"),
        aliases=("pdc", "plasmacytoid dc"),
    ),
    "conventional dendritic cell 1": MarkerProfile(
        canonical_label="conventional dendritic cell 1",
        positive_markers=("CLEC9A", "CADM1", "XCR1", "BTLA", "THBD", "BATF3", "IRF8", "DPP4"),
        negative_markers=("CD1C", "CLEC10A", "FCN1", "S100A8", "CLEC4C"),
        confusion_labels=("dendritic cell", "conventional dendritic cell 2", "monocyte-derived dendritic cell"),
        aliases=("cdc1", "cd141+ dendritic cell", "bdca3+ dendritic cell", "conventional dc1"),
    ),
    "conventional dendritic cell 2": MarkerProfile(
        canonical_label="conventional dendritic cell 2",
        positive_markers=("CD1C", "CLEC10A", "FCER1A", "SIRPA", "CST3", "HLA-DRA", "FCGR2A", "CD2"),
        negative_markers=("CLEC9A", "XCR1", "CLEC4C", "S100A8", "S100A9"),
        confusion_labels=("dendritic cell", "conventional dendritic cell 1", "monocyte-derived dendritic cell", "monocyte"),
        aliases=("cdc2", "cd1c+ dendritic cell", "conventional dc2"),
    ),
    "dendritic cell": MarkerProfile(
        canonical_label="dendritic cell",
        positive_markers=("FCER1A", "HLA-DRA", "HLA-DPA1", "HLA-DPB1", "CD74", "CST3"),
        negative_markers=("S100A8", "S100A9", "C1QA", "C1QB", "MS4A1"),
        confusion_labels=("plasmacytoid dendritic cell", "conventional dendritic cell 1", "conventional dendritic cell 2", "monocyte", "macrophage"),
        aliases=("myeloid dendritic cell",),
    ),
    "monocyte-derived dendritic cell": MarkerProfile(
        canonical_label="monocyte-derived dendritic cell",
        positive_markers=("CD1C", "FCER1A", "CTSS", "LST1", "S100A8", "S100A9", "FCN1", "CLEC10A"),
        negative_markers=("CLEC4C", "CLEC9A", "XCR1", "C1QA"),
        confusion_labels=("dendritic cell", "conventional dendritic cell 2", "monocyte", "macrophage"),
        aliases=("mo-dc", "inflammatory dendritic cell"),
    ),
    "t cell": MarkerProfile(
        canonical_label="t cell",
        positive_markers=("CD3D", "CD3E", "TRBC1", "TRBC2", "IL7R", "LTB", "MAL", "TRAC", "CD247", "TCF7"),
        negative_markers=("NKG7", "GNLY", "MS4A1", "CD79A", "FCER1A"),
        confusion_labels=("naive t cell", "activated t cell", "regulatory t cell", "gamma-delta t cell", "nk cell", "b cell"),
        ontology_ids=("CL:0000084", "CL:0000624"),
        aliases=("cd4-positive, alpha-beta t cell", "cd8-positive, alpha-beta t cell", "t cell"),
    ),
    "naive t cell": MarkerProfile(
        canonical_label="naive t cell",
        positive_markers=("CCR7", "IL7R", "LTB", "MAL", "TCF7"),
        negative_markers=("IFNG", "IL4", "IL5", "GZMB", "NKG7", "FOXP3"),
        confusion_labels=("t cell", "activated t cell", "regulatory t cell"),
        aliases=("naive thymus-derived cd4-positive, alpha-beta t cell", "naive thymus-derived cd8-positive, alpha-beta t cell"),
    ),
    "activated t cell": MarkerProfile(
        canonical_label="activated t cell",
        positive_markers=("CD69", "IL2RA", "MKI67", "CD38", "HLA-DRA", "IFNG"),
        negative_markers=("TCF7", "LTB", "MAL", "FOXP3"),
        confusion_labels=("t cell", "naive t cell", "regulatory t cell"),
        aliases=("activated cd4-positive, alpha-beta t cell", "activated cd8-positive, alpha-beta t cell"),
    ),
    "regulatory t cell": MarkerProfile(
        canonical_label="regulatory t cell",
        positive_markers=("FOXP3", "IL2RA", "CTLA4", "TNFRSF18", "TIGIT"),
        negative_markers=("NKG7", "GNLY", "FCGR3A", "S100A8"),
        confusion_labels=("t cell", "activated t cell"),
        aliases=("treg", "treg cell", "regulatory t cell"),
    ),
    "gamma-delta t cell": MarkerProfile(
        canonical_label="gamma-delta t cell",
        positive_markers=("TRDC", "TRGC1", "TRGC2"),
        negative_markers=("TRBC1", "TRBC2", "MS4A1"),
        confusion_labels=("t cell", "nk cell"),
        aliases=("gamma delta t cell", "γδ t cell"),
    ),
    "progenitor exhausted t cell": MarkerProfile(
        canonical_label="progenitor exhausted t cell",
        positive_markers=("PDCD1", "TCF7", "SLAMF6", "CXCR5", "TOX", "BCL6"),
        negative_markers=("HAVCR2", "GZMB"),
        confusion_labels=("terminal exhausted t cell", "t cell"),
        aliases=("tpex", "precursor exhausted t cell", "progenitor exhausted t cell"),
    ),
    "terminal exhausted t cell": MarkerProfile(
        canonical_label="terminal exhausted t cell",
        positive_markers=("PDCD1", "HAVCR2", "TOX", "GZMB"),
        negative_markers=("TCF7",),
        confusion_labels=("progenitor exhausted t cell", "t cell"),
        aliases=("terminally differentiated exhausted t cell",),
    ),
    "nk cell": MarkerProfile(
        canonical_label="nk cell",
        positive_markers=("NKG7", "GNLY", "PRF1", "GZMB", "KLRD1", "FCGR3A", "CTSW", "TYROBP", "CST7"),
        negative_markers=("CD3D", "CD3E", "TRBC1", "TRBC2", "MS4A1", "CD79A"),
        confusion_labels=("t cell", "gamma-delta t cell"),
        aliases=("natural killer cell", "mature nk t cell"),
    ),
    "b cell": MarkerProfile(
        canonical_label="b cell",
        positive_markers=("MS4A1", "CD79A", "CD79B", "CD74", "HLA-DRA", "BANK1", "CD37", "CD22", "FCRL2", "HVCN1"),
        negative_markers=("CD3D", "TRBC1", "NKG7", "GNLY", "JCHAIN"),
        confusion_labels=("plasma cell", "t cell"),
        aliases=("b cell",),
    ),
    "plasma cell": MarkerProfile(
        canonical_label="plasma cell",
        positive_markers=("JCHAIN", "MZB1", "SDC1", "IGHA1", "IGHA2", "IGHG1", "IGKC", "IGLC2", "XBP1", "TENT5C"),
        negative_markers=("MS4A1", "CD79A", "KIT", "TPSAB1"),
        confusion_labels=("b cell", "mast cell"),
        ontology_ids=("CL:0000786",),
    ),
    "erythrocyte": MarkerProfile(
        canonical_label="erythrocyte",
        positive_markers=("HBB", "HBA1", "HBA2", "ALAS2", "GYPC", "EPB42", "SLC25A37", "AHSP", "BLVRB", "BPGM"),
        negative_markers=("PTPRC", "EPCAM", "PECAM1", "COL1A1"),
        confusion_labels=(),
        ontology_ids=("CL:0000232",),
    ),
    "platelet": MarkerProfile(
        canonical_label="platelet",
        positive_markers=("PF4", "PPBP", "SPARC", "TUBB1", "GP9", "ITGA2B"),
        negative_markers=("PTPRC", "EPCAM", "COL1A1"),
        confusion_labels=("erythrocyte",),
        aliases=("thrombocyte",),
    ),
    "fibroblast": MarkerProfile(
        canonical_label="fibroblast",
        positive_markers=("DCN", "CFD", "COL1A1", "COL1A2", "COL3A1", "LUM", "COL6A1", "COL6A2", "COL6A3", "MGP"),
        negative_markers=("RGS5", "CSPG4", "PECAM1", "VWF", "EPCAM"),
        confusion_labels=("myofibroblast cell", "pericyte", "smooth muscle cell", "endothelial cell", "adipocyte"),
        ontology_ids=("CL:0000057",),
        aliases=(
            "cd34+ fibroblasts",
            "alveolar adventitial fibroblast",
            "fibroblast of cardiac tissue",
            "adventitial cell",
            "connective tissue cell",
            "stromal cell",
            "stromal cell of ovary",
            "pancreatic stellate cell",
            "keratocyte",
        ),
    ),
    "myofibroblast cell": MarkerProfile(
        canonical_label="myofibroblast cell",
        positive_markers=("ACTA2", "TAGLN", "COL1A1", "COL1A2", "THY1", "POSTN"),
        negative_markers=("PECAM1", "VWF", "MS4A1", "CD3D"),
        confusion_labels=("fibroblast", "smooth muscle cell"),
        aliases=("myofibroblast",),
    ),
    "adipocyte": MarkerProfile(
        canonical_label="adipocyte",
        positive_markers=("ADIPOQ", "PLIN1", "FABP4", "LIPE", "DGAT2", "LEP"),
        negative_markers=("COL1A1", "ACTA2", "PECAM1", "PTPRC"),
        confusion_labels=("fibroblast", "mesothelial cell"),
        aliases=("fat cell",),
    ),
    "pericyte": MarkerProfile(
        canonical_label="pericyte",
        positive_markers=("RGS5", "MCAM", "CSPG4", "NOTCH3", "PDGFRB", "ACTA2", "TAGLN", "MYH11", "DES", "COL4A1"),
        negative_markers=("DCN", "COL1A1", "PECAM1", "VWF", "EPCAM"),
        confusion_labels=("fibroblast", "smooth muscle cell", "endothelial cell"),
        ontology_ids=("CL:0000669",),
    ),
    "smooth muscle cell": MarkerProfile(
        canonical_label="smooth muscle cell",
        positive_markers=("ACTA2", "TAGLN", "MYH11", "CNN1", "DES", "MYLK", "CALD1", "TPM2", "COL4A1"),
        negative_markers=("DCN", "COL1A1", "PECAM1", "VWF", "EPCAM"),
        confusion_labels=("pericyte", "fibroblast"),
        ontology_ids=("CL:0000192",),
        aliases=("blood vessel smooth muscle cell", "bronchial smooth muscle cell", "vascular associated smooth muscle cell"),
    ),
    "endothelial cell": MarkerProfile(
        canonical_label="endothelial cell",
        positive_markers=("PECAM1", "VWF", "KDR", "EMCN", "CLDN5", "ESAM", "RAMP2", "PLVAP", "CDH5", "ENG"),
        negative_markers=("COL1A1", "DCN", "ACTA2", "TAGLN", "EPCAM"),
        confusion_labels=("pericyte", "fibroblast", "smooth muscle cell"),
        aliases=(
            "vein endothelial cell",
            "capillary endothelial cell",
            "vascular endothelial cell",
            "cardiac endothelial cell",
            "retinal blood vessel endothelial cell",
            "endothelial cell of arteriole",
            "endothelial cell of artery",
            "endothelial cell of lymphatic vessel",
            "endothelial cell of venule",
        ),
        ontology_ids=("CL:0000115", "CL:0002543"),
    ),
    "mesothelial cell": MarkerProfile(
        canonical_label="mesothelial cell",
        positive_markers=("MSLN", "UPK3B", "KRT19", "WT1", "KRT18"),
        negative_markers=("PECAM1", "VWF", "COL1A1", "ACTA2"),
        confusion_labels=("epithelial cell", "fibroblast", "adipocyte"),
        aliases=("mesothelium cell",),
    ),
    "epithelial cell": MarkerProfile(
        canonical_label="epithelial cell",
        positive_markers=("EPCAM", "KRT8", "KRT18", "KRT19", "KRT17", "MUC1", "KRT7", "KRT20", "CLDN4", "TACSTD2"),
        negative_markers=("COL1A1", "DCN", "PECAM1", "VWF", "PTPRC"),
        confusion_labels=("basal cell", "club cell", "ciliated epithelial cell", "goblet cell", "ductal cell", "retinal pigment epithelial cell"),
        aliases=(
            "glandular secretory epithelial cell",
            "conjunctival epithelial cell",
            "corneal epithelial cell",
            "bladder urothelial cell",
            "ovarian surface epithelial cell",
            "luminal cell of prostate epithelium",
            "basal cell of prostate epithelium",
            "basal cell",
            "serous cell of epithelium of bronchus",
            "serous cell of epithelium of trachea",
        ),
    ),
    "ciliated epithelial cell": MarkerProfile(
        canonical_label="ciliated epithelial cell",
        positive_markers=("FOXJ1", "PIFO", "TPPP3", "DNAH5", "TEKT1"),
        negative_markers=("SCGB1A1", "MUC2", "EPCAM"),
        confusion_labels=("club cell", "pulmonary ionocyte", "goblet cell", "epithelial cell"),
        aliases=("lung multiciliated epithelial cell", "multiciliated columnar cell of tracheobronchial tree", "ciliated cell"),
    ),
    "club cell": MarkerProfile(
        canonical_label="club cell",
        positive_markers=("SCGB1A1", "SCGB3A1", "CYP2F1", "KRT19", "KRT8"),
        negative_markers=("FOXJ1", "MUC2", "SFTPC", "FOXI1"),
        confusion_labels=("ciliated epithelial cell", "goblet cell", "pulmonary ionocyte", "epithelial cell"),
        aliases=("club cell of airway",),
    ),
    "pulmonary alveolar type 1 cell": MarkerProfile(
        canonical_label="pulmonary alveolar type 1 cell",
        positive_markers=("AGER", "HOPX", "CAV1", "CLIC5", "RTKN2"),
        negative_markers=("SFTPC", "SFTPB", "SCGB1A1"),
        confusion_labels=("pulmonary alveolar type 2 cell", "epithelial cell"),
        aliases=("alveolar type 1 cell", "at1 cell"),
    ),
    "pulmonary alveolar type 2 cell": MarkerProfile(
        canonical_label="pulmonary alveolar type 2 cell",
        positive_markers=("SFTPC", "SFTPB", "SFTPA1", "SFTPA2", "ABCA3", "NAPSA"),
        negative_markers=("AGER", "HOPX", "SCGB1A1"),
        confusion_labels=("pulmonary alveolar type 1 cell", "club cell", "epithelial cell"),
        aliases=("alveolar type 2 cell", "at2 cell"),
    ),
    "pulmonary ionocyte": MarkerProfile(
        canonical_label="pulmonary ionocyte",
        positive_markers=("FOXI1", "CFTR", "ATP6V1B1", "ATP6V0D2", "ASCL3"),
        negative_markers=("SCGB1A1", "FOXJ1", "SFTPC"),
        confusion_labels=("club cell", "ciliated epithelial cell", "epithelial cell"),
        aliases=("ionocyte",),
    ),
    "goblet cell": MarkerProfile(
        canonical_label="goblet cell",
        positive_markers=("MUC2", "SPINK4", "TFF3", "REG4", "CLCA1", "AGR2", "FCGBP", "BPIFB1", "KLF4", "AQP8"),
        negative_markers=("ALPI", "FABP1", "CHGA", "FOXJ1", "SCGB1A1"),
        confusion_labels=("enterocyte", "epithelial cell", "ciliated epithelial cell", "club cell"),
        aliases=("respiratory tract goblet cell", "tracheal goblet cell", "small intestine goblet cell", "mucus secreting cell"),
    ),
    "enterocyte": MarkerProfile(
        canonical_label="enterocyte",
        positive_markers=("ALPI", "FABP1", "FABP2", "APOA1", "APOA4", "KRT20", "SI", "DPEP1", "SLC5A1", "ANPEP"),
        negative_markers=("MUC2", "CHGA", "DEFA5", "POU2F3"),
        confusion_labels=("goblet cell", "paneth cell", "intestinal tuft cell", "enteroendocrine cell", "intestinal stem cell"),
        aliases=(
            "best4+ enterocyte",
            "enterocyte of epithelium proper of duodenum",
            "enterocyte of epithelium proper of ileum",
            "enterocyte of epithelium proper of jejunum",
            "enterocyte of epithelium proper of small intestine",
        ),
    ),
    "paneth cell": MarkerProfile(
        canonical_label="paneth cell",
        positive_markers=("DEFA5", "DEFA6", "LYZ", "REG3A", "PLA2G2A"),
        negative_markers=("ALPI", "MUC2", "CHGA"),
        confusion_labels=("enterocyte", "intestinal stem cell", "enteroendocrine cell"),
        aliases=("paneth cell of epithelium of small intestine",),
    ),
    "intestinal tuft cell": MarkerProfile(
        canonical_label="intestinal tuft cell",
        positive_markers=("POU2F3", "TRPM5", "IL25", "AVIL", "GFI1B"),
        negative_markers=("ALPI", "MUC2", "CHGA"),
        confusion_labels=("enterocyte", "goblet cell", "enteroendocrine cell"),
        aliases=("tuft cell", "intestinal tuft cell"),
    ),
    "intestinal stem cell": MarkerProfile(
        canonical_label="intestinal stem cell",
        positive_markers=("LGR5", "OLFM4", "ASCL2", "SMOC2", "RGMB"),
        negative_markers=("ALPI", "MUC2", "CHGA", "DEFA5"),
        confusion_labels=("transit amplifying cell", "enterocyte", "paneth cell"),
        aliases=("intestinal crypt stem cell of small intestine", "stem cell of small intestine"),
    ),
    "transit amplifying cell": MarkerProfile(
        canonical_label="transit amplifying cell",
        positive_markers=("MKI67", "TOP2A", "BIRC5", "STMN1", "PCNA"),
        negative_markers=("ALPI", "MUC2", "CHGA", "DEFA5"),
        confusion_labels=("intestinal stem cell", "enterocyte", "goblet cell"),
        aliases=("transit amplifying cell of small intestine",),
    ),
    "enteroendocrine cell": MarkerProfile(
        canonical_label="enteroendocrine cell",
        positive_markers=("CHGA", "CHGB", "NEUROD1", "PAX6", "TPH1", "PCSK1", "SCG2", "SCG3", "INSM1", "NKX2-2"),
        negative_markers=("ALPI", "MUC2", "DEFA5", "POU2F3"),
        confusion_labels=("goblet cell", "enterocyte", "paneth cell"),
        aliases=("enteroendocrine cell of small intestine",),
    ),
    "ductal cell": MarkerProfile(
        canonical_label="ductal cell",
        positive_markers=("KRT19", "KRT8", "KRT18", "MUC1", "TFF1", "SLC4A4", "KRT17", "SOX9", "ANXA4", "EPCAM"),
        negative_markers=("PRSS1", "CPA1", "CTRB1", "INS", "GCG"),
        confusion_labels=("acinar cell", "epithelial cell"),
        aliases=("duct epithelial cell", "pancreatic ductal cell"),
    ),
    "acinar cell": MarkerProfile(
        canonical_label="acinar cell",
        positive_markers=("PRSS1", "PRSS2", "CPA1", "CTRB1", "CTRB2", "CELA3A", "CELA3B", "REG1A", "REG1B", "PNLIP"),
        negative_markers=("KRT19", "MUC1", "SOX9", "INS"),
        confusion_labels=("ductal cell",),
        aliases=("pancreatic acinar cell", "acinar cell of salivary gland"),
    ),
    "pancreatic beta cell": MarkerProfile(
        canonical_label="pancreatic beta cell",
        positive_markers=("INS", "IAPP", "PCSK1", "CPE", "PDX1", "MAFA", "NKX6-1", "ABCC8", "SLC30A8", "CHGA"),
        negative_markers=("GCG", "SST", "PPY"),
        confusion_labels=("pancreatic alpha cell", "pancreatic delta cell", "pancreatic pp cell"),
        aliases=("beta cell", "type b pancreatic cell"),
    ),
    "pancreatic alpha cell": MarkerProfile(
        canonical_label="pancreatic alpha cell",
        positive_markers=("GCG", "TTR", "ARX", "IRX2", "LOXL4", "CHGA", "PCSK2", "ISL1", "GC", "TM4SF4"),
        negative_markers=("INS", "SST", "PPY"),
        confusion_labels=("pancreatic beta cell", "pancreatic delta cell"),
        aliases=("alpha cell", "pancreatic a cell"),
    ),
    "pancreatic delta cell": MarkerProfile(
        canonical_label="pancreatic delta cell",
        positive_markers=("SST", "HHEX", "RBP4", "PCSK2", "CHGA", "GHSR", "RGS4", "GABRA5", "GC"),
        negative_markers=("INS", "GCG", "PPY"),
        confusion_labels=("pancreatic beta cell", "pancreatic alpha cell"),
        aliases=("delta cell", "pancreatic d cell"),
    ),
    "pancreatic pp cell": MarkerProfile(
        canonical_label="pancreatic pp cell",
        positive_markers=("PPY", "PAX6", "PCSK2", "CHGA", "TTR", "SCG5", "PAM", "GC", "SLC18A1", "ISL1"),
        negative_markers=("INS", "GCG", "SST"),
        confusion_labels=("pancreatic beta cell", "pancreatic alpha cell", "pancreatic delta cell"),
        aliases=("pp cell", "pancreatic pp cell"),
    ),
    "cardiomyocyte": MarkerProfile(
        canonical_label="cardiomyocyte",
        positive_markers=("TNNT2", "MYH6", "MYH7", "ACTC1", "TTN", "RYR2", "PLN", "TNNI3", "MYBPC3", "NKX2-5"),
        negative_markers=("ACTA2", "COL1A1", "PECAM1"),
        confusion_labels=("smooth muscle cell", "fibroblast"),
        aliases=("regular atrial cardiac myocyte", "ventricular cardiac muscle cell"),
    ),
    "glial cell": MarkerProfile(
        canonical_label="glial cell",
        positive_markers=("S100B", "GLUL", "SOX9", "FABP7", "VIM"),
        negative_markers=("SNAP25", "STMN2", "PTPRC"),
        confusion_labels=("microglial cell", "schwann cell", "neuron"),
        aliases=("enteroglial cell", "radial glial cell", "mueller cell"),
    ),
    "schwann cell": MarkerProfile(
        canonical_label="schwann cell",
        positive_markers=("SOX10", "S100B", "MPZ", "PMP22", "PLP1"),
        negative_markers=("SNAP25", "STMN2", "TMEM119"),
        confusion_labels=("glial cell", "neuron"),
        aliases=("schwann cell",),
    ),
    "neuron": MarkerProfile(
        canonical_label="neuron",
        positive_markers=("SNAP25", "STMN2", "TUBB3", "MAP2", "ELAVL4"),
        negative_markers=("S100B", "TMEM119", "COL1A1"),
        confusion_labels=("glial cell",),
        aliases=("retinal bipolar neuron", "retinal ganglion cell", "eye photoreceptor cell", "neuron"),
    ),
    "retinal pigment epithelial cell": MarkerProfile(
        canonical_label="retinal pigment epithelial cell",
        positive_markers=("RPE65", "BEST1", "RLBP1", "PMEL", "MITF"),
        negative_markers=("SNAP25", "STMN2", "TMEM119"),
        confusion_labels=("epithelial cell", "glial cell"),
        aliases=("rpe cell",),
    ),
}


def _normalize_label_key(value: str) -> str:
    return " ".join(value.strip().lower().replace("_", " ").split())


LABEL_ALIASES = {}
ONTOLOGY_TO_LABEL = {}
for canonical_label, profile in MARKER_PROFILES.items():
    LABEL_ALIASES[_normalize_label_key(canonical_label)] = canonical_label
    for alias in profile.aliases:
        LABEL_ALIASES[_normalize_label_key(alias)] = canonical_label
    for ontology_id in profile.ontology_ids:
        ONTOLOGY_TO_LABEL[ontology_id] = canonical_label


def canonicalize_label(label: str | None, metadata: dict | None = None) -> str | None:
    if metadata:
        ontology_id = metadata.get("cell_type_ontology_term_id")
        if ontology_id and ontology_id in ONTOLOGY_TO_LABEL:
            return ONTOLOGY_TO_LABEL[ontology_id]

    candidates = []
    if label:
        candidates.append(label)
    if metadata:
        for key in ("cell_type", "free_annotation", "broad_cell_class"):
            value = metadata.get(key)
            if value:
                candidates.append(value)

    for candidate in candidates:
        normalized = _normalize_label_key(candidate)
        if normalized in LABEL_ALIASES:
            return LABEL_ALIASES[normalized]
    return None


def get_marker_profile(label: str | None, metadata: dict | None = None) -> MarkerProfile | None:
    canonical_label = canonicalize_label(label, metadata)
    if not canonical_label:
        return None
    return MARKER_PROFILES.get(canonical_label)


def filter_informative_markers(genes: list[str]) -> list[str]:
    filtered = []
    for gene in genes:
        upper = gene.upper()
        if upper.startswith(("RPL", "RPS", "MT-", "MTRNR", "HSP", "LINC")):
            continue
        if upper in GENERIC_MARKERS:
            continue
        filtered.append(gene)
    return filtered


def observed_positive_markers(genes: list[str], label: str | None, metadata: dict | None = None) -> list[str]:
    profile = get_marker_profile(label, metadata)
    if profile is None:
        return []

    observed = {gene.upper() for gene in genes}
    return [marker for marker in profile.positive_markers if marker.upper() in observed]


def supported_markers(genes: list[str], label: str | None, metadata: dict | None = None) -> list[tuple[str, float]]:
    profile = get_marker_profile(label, metadata)
    if profile is None:
        return []

    informative = set(filter_informative_markers(genes))
    ranks = {gene.upper(): index for index, gene in enumerate(genes)}
    matches = []
    total = max(1, len(genes))
    for marker in profile.positive_markers:
        if marker not in informative:
            continue
        rank = ranks.get(marker.upper())
        if rank is None:
            continue
        score = round(1.0 - (rank / total), 4)
        matches.append((marker, score))

    matches.sort(key=lambda item: (-item[1], item[0]))
    return matches


def candidate_labels_for_gold(gold_label: str | None, metadata: dict | None = None) -> list[str]:
    profile = get_marker_profile(gold_label, metadata)
    if profile is None:
        return []
    return list(profile.confusion_labels)


def top_missing_markers(genes: list[str], label: str | None, metadata: dict | None = None, count: int = 2) -> list[str]:
    profile = get_marker_profile(label, metadata)
    if profile is None:
        return []
    observed = {gene.upper() for gene in genes}
    missing = []
    for marker in profile.positive_markers:
        if marker.upper() in observed:
            continue
        if marker.upper() in GENERIC_MARKERS:
            continue
        missing.append(marker)
        if len(missing) >= count:
            break
    return missing