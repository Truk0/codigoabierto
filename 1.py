import zipfile
import posixpath
import pandas as pd
from lxml import etree
from collections import defaultdict

def analizar_pptx(ruta_pptx, salida_excel="pesos_slides.xlsx"):

    ns = {
        "r": "http://schemas.openxmlformats.org/package/2006/relationships"
    }

    resultados = []
    uso_recursos = defaultdict(set)

    with zipfile.ZipFile(ruta_pptx, "r") as pptx:

        archivos = set(pptx.namelist())

        slides = sorted(
            [
                f for f in archivos
                if f.startswith("ppt/slides/slide")
                and f.endswith(".xml")
                and "/" not in f[len("ppt/slides/"):]
            ],
            key=lambda x: int(
                x.split("slide")[-1].split(".")[0]
            )
        )

        def obtener_dependencias(archivo, visitados=None):
            if visitados is None:
                visitados = set()

            if archivo in visitados or archivo not in archivos:
                return set()

            visitados.add(archivo)
            dependencias = {archivo}

            carpeta = posixpath.dirname(archivo)
            nombre = posixpath.basename(archivo)

            rels = posixpath.join(
                carpeta, "_rels", nombre + ".rels"
            )

            if rels in archivos:
                dependencias.add(rels)

                raiz = etree.fromstring(pptx.read(rels))

                for relacion in raiz:
                    if relacion.get("TargetMode") == "External":
                        continue

                    target = relacion.get("Target")

                    if not target:
                        continue

                    if target.startswith("/"):
                        destino = target.lstrip("/")
                    else:
                        destino = posixpath.normpath(
                            posixpath.join(carpeta, target)
                        )

                    dependencias.update(
                        obtener_dependencias(destino, visitados)
                    )

            return dependencias

        dependencias_slides = {}

        for slide in slides:
            deps = obtener_dependencias(slide)
            dependencias_slides[slide] = deps

            for recurso in deps:
                uso_recursos[recurso].add(slide)

        for slide in slides:

            deps = dependencias_slides[slide]

            peso_total = sum(
                pptx.getinfo(f).file_size for f in deps
            )

            peso_comprimido = sum(
                pptx.getinfo(f).compress_size for f in deps
            )

            exclusivos = {
                f for f in deps
                if len(uso_recursos[f]) == 1
            }

            peso_exclusivo = sum(
                pptx.getinfo(f).file_size
                for f in exclusivos
            )

            numero = int(
                slide.split("slide")[-1].split(".")[0]
            )

            resultados.append({
                "Diapositiva": numero,
                "Peso independiente (MB)": round(
                    peso_total / 1024**2, 3
                ),
                "Peso comprimido (MB)": round(
                    peso_comprimido / 1024**2, 3
                ),
                "Peso exclusivo (MB)": round(
                    peso_exclusivo / 1024**2, 3
                ),
                "Número de recursos": len(deps)
            })

    df = pd.DataFrame(resultados)

    df.to_excel(salida_excel, index=False)

    return df


# Ejemplo
df = analizar_pptx("presentacion.pptx")
print(df)
