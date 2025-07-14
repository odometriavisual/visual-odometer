import numpy as np
import matplotlib.pyplot as plt


def create_pattern_image(h, w):
    image = np.full((h, w), 100, dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            image[i, j] = int(100 + 50 * np.sin((i + j) * 0.08))

    center_y, center_x = h // 2, w // 2
    for i in range(h):
        for j in range(w):
            dist = np.sqrt((i - center_y) ** 2 + (j - center_x) ** 2)
            if int(dist) % 12 < 3:  # Anéis a cada 12 pixels
                image[i, j] = 255

    for i in range(0, min(15, h)):
        for j in range(0, min(i + 3, w)):
            image[i, j] = 50

    for i in range(max(0, h - 15), h):
        for j in range(max(0, w - (h - i) - 3), w):
            image[i, j] = 200

    for j in range(0, min(8, w)):
        if j % 2 == 0:
            image[:, j] = 30

    for j in range(max(0, w - 8), w):
        if j % 2 == 0:
            image[:, j] = 220

    for i in range(0, min(8, h)):
        if i % 2 == 0:
            image[i, :] = 40

    for i in range(max(0, h - 8), h):
        if i % 2 == 0:
            image[i, :] = 210

    # 6. Marcadores de orientação únicos
    # Triângulo no canto superior esquerdo
    for i in range(6):
        for j in range(6 - i):
            image[i, j] = 255

    # Quadrado no canto superior direito
    image[0:6, w - 6:w] = 0

    # X no canto inferior esquerdo
    for i in range(6):
        image[h - 6 + i, i] = 255
        image[h - 6 + i, 5 - i] = 255

    # Cruz no canto inferior direito
    image[h - 6:h, w - 3] = 0
    image[h - 3, w - 6:w] = 0

    # 7. Padrão em formato de seta apontando para o centro
    # Seta da esquerda
    for i in range(h // 2 - 5, h // 2 + 6):
        for j in range(15):
            if abs(i - h // 2) <= j // 3:
                image[i, j] = 180

    # Seta da direita
    for i in range(h // 2 - 5, h // 2 + 6):
        for j in range(w - 15, w):
            if abs(i - h // 2) <= (w - 1 - j) // 3:
                image[i, j] = 180

    return image


def add_grid(image, grid_spacing=20, grid_value=70, offset=3):
    """Adiciona linhas de grade visuais à imagem."""
    image_with_grid = image.copy()
    h, w = image.shape

    for y in range(offset, h, grid_spacing):
        image_with_grid[y, :] = np.minimum(image_with_grid[y, :], grid_value)
    for x in range(offset, w, grid_spacing):
        image_with_grid[:, x] = np.minimum(image_with_grid[:, x], grid_value)

    return image_with_grid


def generate_shifted_images(h, w, dx, dy):
    """
    Gera imagens A e B onde B é A deslocada.
    CORRIGIDO: Agora corta dois pedaços de uma imagem maior mantendo continuidade.
    """
    # Criar uma imagem maior que contenha ambas as regiões
    # Precisa ser grande o suficiente para conter A e B deslocado
    margin = 50  # margem extra para evitar problemas nas bordas
    large_h = h + abs(dy) + 2 * margin
    large_w = w + abs(dx) + 2 * margin

    # Criar o padrão na imagem grande
    large_pattern = create_pattern_image(large_h, large_w)

    # Definir posição de A na imagem grande (no centro da margem)
    start_y_A = margin
    start_x_A = margin

    # Definir posição de B na imagem grande (A deslocada por dx, dy)
    start_y_B = start_y_A + dy
    start_x_B = start_x_A + dx

    # Extrair imagem A (região base)
    imgA = large_pattern[start_y_A:start_y_A + h, start_x_A:start_x_A + w].copy()

    # Extrair imagem B (região deslocada)
    imgB = large_pattern[start_y_B:start_y_B + h, start_x_B:start_x_B + w].copy()

    return imgA, imgB


def align_images_by_displacement(imgA, imgB, dx, dy, mode='crop', crop_size=(60, 40)):
    """
    Alinha duas imagens deslocadas aplicando metade do deslocamento em cada uma.
    CORRIGIDO: Agora que B = A deslocada por (dx,dy), para alinhar precisamos
    mover A por (+dx/2, +dy/2) e B por (-dx/2, -dy/2)
    """
    h, w = imgA.shape
    dx_abs, dy_abs = abs(dx), abs(dy)

    if mode == 'crop':
        # Corte simples para alinhar deslocamento (A frente, B atrás)
        if dx > 0:
            imgA = imgA[:, dx:]
            imgB = imgB[:, :-dx]
        elif dx < 0:
            imgA = imgA[:, :dx]
            imgB = imgB[:, -dx:]

        if dy > 0:
            imgA = imgA[dy:, :]
            imgB = imgB[:-dy, :]
        elif dy < 0:
            imgA = imgA[:dy, :]
            imgB = imgB[-dy:, :]

        # Garantir tamanhos iguais
        min_h = min(imgA.shape[0], imgB.shape[0])
        min_w = min(imgA.shape[1], imgB.shape[1])
        return imgA[:min_h, :min_w], imgB[:min_h, :min_w]


    elif mode == 'center_crop':
        crop_w, crop_h = crop_size
        cx, cy = w // 2, h // 2

        # CORRIGIDO: Compensação de deslocamento
        # A precisa mover +dx/2, +dy/2 (em direção a B)
        # B precisa mover -dx/2, -dy/2 (em direção a A)
        offset_x_A = dx // 2
        offset_y_A = dy // 2
        offset_x_B = -(dx - offset_x_A)  # resto do deslocamento na direção oposta
        offset_y_B = -(dy - offset_y_A)

        xA1 = cx - crop_w // 2 + offset_x_A
        yA1 = cy - crop_h // 2 + offset_y_A
        xB1 = cx - crop_w // 2 + offset_x_B
        yB1 = cy - crop_h // 2 + offset_y_B

        xA2, yA2 = xA1 + crop_w, yA1 + crop_h
        xB2, yB2 = xB1 + crop_w, yB1 + crop_h

        # Garantir que os crops estão dentro dos limites
        xA1, yA1 = max(0, xA1), max(0, yA1)
        xA2, yA2 = min(w, xA2), min(h, yA2)
        xB1, yB1 = max(0, xB1), max(0, yB1)
        xB2, yB2 = min(w, xB2), min(h, yB2)

        imgA_crop = imgA[yA1:yA2, xA1:xA2]
        imgB_crop = imgB[yB1:yB2, xB1:xB2]

        return imgA_crop, imgB_crop

    elif mode == 'pad_mean':
        def pad_symmetric_corrected(img, dx, dy, direction, mean_val):
            h, w = img.shape

            if direction == 'B':
                dx_offset = dx // 2
                dy_offset = dy // 2

                if dx_offset > 0:
                    img = img[:, :w - dx_offset]  # crop direita
                    img = np.pad(img, ((0, 0), (dx_offset, 0)), constant_values=mean_val)  # pad esquerda
                elif dx_offset < 0:
                    img = img[:, -dx_offset:]  # crop esquerda
                    img = np.pad(img, ((0, 0), (0, -dx_offset)), constant_values=mean_val)  # pad direita

                if dy_offset > 0:
                    img = img[:h - dy_offset, :]  # crop fundo
                    img = np.pad(img, ((dy_offset, 0), (0, 0)), constant_values=mean_val)  # pad topo
                elif dy_offset < 0:
                    img = img[-dy_offset:, :]  # crop topo
                    img = np.pad(img, ((0, -dy_offset), (0, 0)), constant_values=mean_val)  # pad fundo

            else:  # direction == 'B'
                dx_offset = -(dx - dx // 2)
                dy_offset = -(dy - dy // 2)

                if dx_offset > 0:
                    img = img[:, :w - dx_offset]  # crop direita
                    img = np.pad(img, ((0, 0), (dx_offset, 0)), constant_values=mean_val)  # pad esquerda
                elif dx_offset < 0:
                    img = img[:, -dx_offset:]  # crop esquerda
                    img = np.pad(img, ((0, 0), (0, -dx_offset)), constant_values=mean_val)  # pad direita

                # Se dy_offset > 0: B move para baixo, então pad topo e crop fundo
                if dy_offset > 0:
                    img = img[:h - dy_offset, :]  # crop fundo
                    img = np.pad(img, ((dy_offset, 0), (0, 0)), constant_values=mean_val)  # pad topo
                elif dy_offset < 0:
                    img = img[-dy_offset:, :]  # crop topo
                    img = np.pad(img, ((0, -dy_offset), (0, 0)), constant_values=mean_val)  # pad fundo

            return img

        meanA = int(np.mean(imgA))
        meanB = int(np.mean(imgB))

        imgA_padded = pad_symmetric_corrected(imgA, dx, dy, 'A', meanA)
        imgB_padded = pad_symmetric_corrected(imgB, dx, dy, 'B', meanB)

        # Garantir mesmo tamanho final
        min_h = min(imgA_padded.shape[0], imgB_padded.shape[0])
        min_w = min(imgA_padded.shape[1], imgB_padded.shape[1])
        return imgA_padded[:min_h, :min_w], imgB_padded[:min_h, :min_w]

    elif mode == 'wrap':
        def wrap_symmetric(img, dx, dy, direction):
            h, w = img.shape

            dx_half = abs(dx) // 2
            dy_half = abs(dy) // 2

            # Eixo X
            if dx != 0:
                if direction == 'B' and dx > 0:
                    left = img[:, -dx_half:]  # pegar da direita
                    core = img[:, :w - dx_half]
                    img = np.concatenate([left, core], axis=1)
                elif direction == 'A' and dx > 0:
                    core = img[:, dx_half:]
                    right = img[:, :dx_half]
                    img = np.concatenate([core, right], axis=1)
                elif direction == 'B' and dx < 0:
                    right = img[:, :dx_half]
                    core = img[:, dx_half:]
                    img = np.concatenate([core, right], axis=1)
                elif direction == 'A' and dx < 0:
                    core = img[:, :w - dx_half]
                    left = img[:, -dx_half:]
                    img = np.concatenate([left, core], axis=1)

            # Eixo Y
            if dy != 0:
                if direction == 'B' and dy > 0:
                    top = img[-dy_half:, :]
                    core = img[:h - dy_half, :]
                    img = np.concatenate([top, core], axis=0)
                elif direction == 'A' and dy > 0:
                    core = img[dy_half:, :]
                    bottom = img[:dy_half, :]
                    img = np.concatenate([core, bottom], axis=0)
                elif direction == 'B' and dy < 0:
                    bottom = img[:dy_half, :]
                    core = img[dy_half:, :]
                    img = np.concatenate([core, bottom], axis=0)
                elif direction == 'A' and dy < 0:
                    core = img[:h - dy_half, :]
                    top = img[-dy_half:, :]
                    img = np.concatenate([top, core], axis=0)

            return img

        imgA_wrap = wrap_symmetric(imgA, dx, dy, 'A')
        imgB_wrap = wrap_symmetric(imgB, dx, dy, 'B')

        # Garantir mesmo tamanho final
        min_h = min(imgA_wrap.shape[0], imgB_wrap.shape[0])
        min_w = min(imgA_wrap.shape[1], imgB_wrap.shape[1])
        return imgA_wrap[:min_h, :min_w], imgB_wrap[:min_h, :min_w]

    else:
        raise ValueError(f"Modo '{mode}' não reconhecido.")


def add_size_text(ax, img, title):
    """Adiciona texto com dimensões da imagem."""
    h, w = img.shape
    ax.text(2, h - 2, f'{w}×{h}px',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            fontsize=8, ha='left', va='bottom')
    ax.set_title(title, fontsize=11)


def visualize_all_methods():
    """Visualiza todos os métodos de alinhamento em uma única figura."""
    # Parâmetros - imagem retangular
    h, w = 240, 360
    dx, dy = 30, 80

    # Gerar imagem A e B deslocada (agora com continuidade correta)
    imgA, imgB = generate_shifted_images(h, w, dx, dy)

    # Adicionar grade
    imgA_grid = add_grid(imgA)
    imgB_grid = add_grid(imgB)

    # Métodos a testar
    methods = ['crop', 'center_crop', 'pad_mean', 'wrap']
    method_names = ['Corte nas Bordas', 'Corte Central', 'Preenchimento c/ Média', 'Wrap Circular']

    # Criar figura
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))

    # Primeira linha: imagens originais
    axes[0, 0].imshow(imgA_grid, cmap='gray', vmin=0, vmax=255)
    add_size_text(axes[0, 0], imgA_grid, 'Imagem A (Original)')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(imgB_grid, cmap='gray', vmin=0, vmax=255)
    add_size_text(axes[0, 1], imgB_grid, 'Imagem B (Deslocada)')
    axes[0, 1].axis('off')

    # Adicionar texto explicativo
    axes[0, 2].text(0.5, 0.5,
                    f'Deslocamento aplicado:\ndx = {dx}px\ndy = {dy}px\n\nCada método aplica\nmetade do deslocamento\nem cada imagem',
                    ha='center', va='center', transform=axes[0, 2].transAxes,
                    fontsize=10)
    axes[0, 2].axis('off')

    # Ocultar o último da primeira linha
    axes[0, 3].axis('off')

    # Segunda e terceira linha: métodos
    for i, (method, name) in enumerate(zip(methods, method_names)):
        try:
            imgA_aligned, imgB_aligned = align_images_by_displacement(
                imgA_grid, imgB_grid, dx, dy, mode=method, crop_size=(80, 60)
            )

            # Segunda linha: Imagem A processada
            axes[1, i].imshow(imgA_aligned, cmap='gray', vmin=0, vmax=255)
            add_size_text(axes[1, i], imgA_aligned, f'A - {name}')
            axes[1, i].axis('off')

            # Terceira linha: Imagem B processada
            axes[2, i].imshow(imgB_aligned, cmap='gray', vmin=0, vmax=255)
            add_size_text(axes[2, i], imgB_aligned, f'B - {name}')
            axes[2, i].axis('off')

        except Exception as e:
            print(f"Erro no método {method}: {e}")

    plt.tight_layout(rect=[0, 0, 1, 1])  # reserva espaço para o suptitle
    plt.suptitle('Métodos de Alinhamento: Recentralização de Imagens Deslocadas',
                 fontsize=16, fontweight='bold', y=0.99)
    plt.savefig("plot.png")
    plt.show()


def demonstrate_single_method(method='wrap'):
    h, w = 100, 140
    dx, dy = 12, 18

    # Gerar imagens
    imgA, imgB = generate_shifted_images(h, w, dx, dy)

    imgA_grid = add_grid(imgA)
    imgB_grid = add_grid(imgB)

    imgA_aligned, imgB_aligned = align_images_by_displacement(
        imgA_grid, imgB_grid, dx, dy, mode=method, crop_size=(80, 60)
    )

    # Plotar
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].imshow(imgA_grid, cmap='gray', vmin=0, vmax=255)
    add_size_text(axes[0, 0], imgA_grid, 'Imagem A - Original')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(imgB_grid, cmap='gray', vmin=0, vmax=255)
    add_size_text(axes[0, 1], imgB_grid, 'Imagem B - Deslocada')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(imgA_aligned, cmap='gray', vmin=0, vmax=255)
    add_size_text(axes[1, 0], imgA_aligned, f'Imagem A - {method}')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(imgB_aligned, cmap='gray', vmin=0, vmax=255)
    add_size_text(axes[1, 1], imgB_aligned, f'Imagem B - {method}')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.suptitle(f'Método: {method} - Recentralização (dx={dx}, dy={dy})', fontsize=14, fontweight='bold')
    plt.show()


def demonstrate_original_concept():
    """Demonstra o conceito original: uma imagem grande com A e B sendo crops dela."""
    h, w = 100, 140
    dx, dy = 12, 18

    # Criar imagem grande
    margin = 50
    large_h = h + abs(dy) + 2 * margin
    large_w = w + abs(dx) + 2 * margin
    large_pattern = create_pattern_image(large_h, large_w)
    large_pattern_grid = add_grid(large_pattern)

    # Posições das regiões A e B
    start_y_A = margin
    start_x_A = margin
    start_y_B = start_y_A + dy
    start_x_B = start_x_A + dx

    # Criar figura
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Imagem grande com retângulos mostrando A e B
    axes[0].imshow(large_pattern_grid, cmap='gray', vmin=0, vmax=255)

    # Desenhar retângulos para mostrar as regiões
    from matplotlib.patches import Rectangle
    rectA = Rectangle((start_x_A, start_y_A), w, h, linewidth=3, edgecolor='red', facecolor='none', label='Região A')
    rectB = Rectangle((start_x_B, start_y_B), w, h, linewidth=3, edgecolor='blue', facecolor='none', label='Região B')
    axes[0].add_patch(rectA)
    axes[0].add_patch(rectB)
    axes[0].set_title('Imagem Grande com Regiões A e B', fontsize=12)
    axes[0].legend()
    axes[0].axis('off')

    # Extrair e mostrar A e B
    imgA = large_pattern_grid[start_y_A:start_y_A + h, start_x_A:start_x_A + w]
    imgB = large_pattern_grid[start_y_B:start_y_B + h, start_x_B:start_x_B + w]

    axes[1].imshow(imgA, cmap='gray', vmin=0, vmax=255)
    add_size_text(axes[1], imgA, 'Imagem A (crop da região vermelha)')
    axes[1].axis('off')

    axes[2].imshow(imgB, cmap='gray', vmin=0, vmax=255)
    add_size_text(axes[2], imgB, 'Imagem B (crop da região azul)')
    axes[2].axis('off')

    plt.tight_layout()
    plt.suptitle('Conceito: B é uma continuação de A da mesma imagem fonte', fontsize=14, fontweight='bold')
    plt.show()


# ---------- Demonstração ----------
print("Executando demonstração do conceito corrigido...")
demonstrate_original_concept()

print("\nExecutando visualização completa...")
visualize_all_methods()

print("\nPara demonstrar um método específico, use:")
print("demonstrate_single_method('wrap')")
print("demonstrate_single_method('crop')")
print("demonstrate_single_method('center_crop')")
print("demonstrate_single_method('pad_mean')")