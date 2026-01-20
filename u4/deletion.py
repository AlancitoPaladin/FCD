from PIL import Image

# --- CONFIGURACIÓN ---
NOMBRE_IMAGEN = "toro.jpeg"
FILAS = 3  # Queremos 3 filas
COLUMNAS = 4  # Queremos 4 columnas -> 3 * 4 = 12 piezas
NOMBRE_SALIDA = "pieza"

try:
    # 1. Abre la imagen
    img = Image.open(NOMBRE_IMAGEN)
except FileNotFoundError:
    print(f"ERROR: No se encontró el archivo '{NOMBRE_IMAGEN}'.")
    # Puedes intentar buscar el nombre si no lo has renombrado:
    # img = Image.open("ruta/a/tu/imagen.jpg")
    exit()

ancho_original, alto_original = img.size

# 2. Calcula el tamaño de cada "baldosa" (tile)
ancho_por_pieza = ancho_original // COLUMNAS
alto_por_pieza = alto_original // FILAS

print(f"Imagen original: {ancho_original}x{alto_original} píxeles")
print(f"Cada pieza será de: {ancho_por_pieza}x{alto_por_pieza} píxeles")

# 3. Recorre la cuadrícula y recorta
contador = 1
for i in range(FILAS):  # Fila (eje Y)
    for j in range(COLUMNAS):  # Columna (eje X)

        # Coordenadas de recorte (left, upper, right, lower)
        # La librería Pillow usa un sistema de coordenadas (x, y) donde (0,0) es la esquina superior izquierda.
        left = j * ancho_por_pieza
        upper = i * alto_por_pieza
        right = left + ancho_por_pieza
        lower = upper + alto_por_pieza

        # Opcional: Ajustar la última columna/fila para capturar píxeles sobrantes
        if j == COLUMNAS - 1:
            right = ancho_original
        if i == FILAS - 1:
            lower = alto_original

        # Recorta la pieza
        pieza = img.crop((left, upper, right, lower))

        # 4. Guarda la pieza recortada
        nombre_archivo = f"{NOMBRE_SALIDA}_{contador}.png"
        pieza.save(nombre_archivo)

        print(f"Guardado: {nombre_archivo}")
        contador += 1

print("\n¡Proceso de subdivisión terminado!")

#