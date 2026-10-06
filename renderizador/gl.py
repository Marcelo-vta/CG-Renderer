#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# pylint: disable=invalid-name

"""
Biblioteca Gráfica / Graphics Library.

Desenvolvido por: Marcelo Alonso e Lucas Abatepietro
Disciplina: Computação Gráfica
Data: 12/08/2026
"""

import time         # Para operações com tempo
import gpu          # Simula os recursos de uma GPU
import math         # Funções matemáticas
import numpy as np  # Biblioteca do Numpy

class GL:
    """Classe que representa a biblioteca gráfica (Graphics Library)."""

    width = 800   # largura da tela
    height = 600  # altura da tela
    near = 0.01   # plano de corte próximo
    far = 1000    # plano de corte distante

    viewpoint_val = np.eye(4) # Matriz de viewpoint (tela @ projeção @ câmera)
    transform_stack = [np.eye(4)] # Pilha de matrizes de transformação

    # Posição e rotação da câmera no mundo, usadas no vetor até o observador e no headlight
    eye_position = np.zeros(3)
    eye_rotation = np.eye(3)

    # Luzes da cena, reiniciadas a cada frame em clear_buffers (o headlight vem do NavigationInfo)
    headlight = False
    lights = []

    # Texturas já lidas, com todos os níveis do mipmap: nome do arquivo -> [nível 0, nível 1, ...]
    texture_cache = {}

    # Instante em que a animação começou (primeira chamada do TimeSensor)
    start_time = None

    # Buffers de supersampling (4 samples por pixel). Os triângulos só escrevem neles; quem
    # converte o resultado em imagem é o render_buffer, uma única vez por frame.
    # Posição (x, y) de cada sample dentro do pixel, no padrão RGSS 2x2 (grade 4x4 com um
    # sample por linha e por coluna)
    offsets = np.array([[0.625, 0.125], [0.125, 0.375], [0.875, 0.625], [0.375, 0.875]])
    depth_buffer = np.full((height, width, 4), np.inf)  # distância até a câmera de cada sample
    color_buffer = np.zeros((height, width, 4, 3))      # cor (0 a 255) de cada sample
    touched = np.zeros((height, width), dtype=bool)     # pixels em que algum sample foi desenhado

    @staticmethod
    def draw(coord, color):

        fb_dim = gpu.GPU.frame_buffer[gpu.GPU.draw_framebuffer].color.shape

        if coord[0] > fb_dim[1]-1:
            return
        if coord[0] < 0:
            return
        if coord[1] > fb_dim[0]-1:
            return
        if coord[1] < 0:
            return

        gpu.GPU.draw_pixel(coord, gpu.GPU.RGB8, color)

    @staticmethod
    def setup(width, height, near=0.01, far=1000):
        """Definr parametros para câmera de razão de aspecto, plano próximo e distante."""
        GL.width = width
        GL.height = height
        GL.near = near
        GL.far = far

        GL.depth_buffer = np.full((height, width, 4), np.inf)
        GL.color_buffer = np.zeros((height, width, 4, 3))
        GL.touched = np.zeros((height, width), dtype=bool)
        GL.start_time = None

    @staticmethod
    def clear_buffers():
        """Reinicia os buffers de supersampling e as luzes (chamar uma vez no início de cada frame)."""
        GL.depth_buffer.fill(np.inf)                  # nenhum sample foi coberto ainda
        GL.color_buffer[:] = gpu.GPU.clear_color_val  # samples começam com a cor de fundo
        GL.touched.fill(False)

        # As luzes são lidas de novo a cada frame, pois o traversal do grafo de cena as repete
        GL.lights = []
        GL.headlight = False

    @staticmethod
    def render_buffer():
        """Renderiza o buffer de supersampling no framebuffer (chamar uma vez no fim de cada frame)."""
        fb = gpu.GPU.frame_buffer[gpu.GPU.draw_framebuffer].color

        # Só mexe nos pixels em que algum sample foi desenhado, assim o que já estava no
        # framebuffer (fundo, desenhos 2D feitos direto com GL.draw) é preservado
        covered = GL.touched
        if not covered.any():
            return

        # Cor do pixel = média da cor dos seus samples (os não cobertos têm a cor de fundo)
        pixels = GL.color_buffer[covered].mean(axis=1)
        fb[covered, :3] = np.clip(np.rint(pixels), 0, 255).astype(np.uint8)



    @staticmethod
    def polypoint2D(point, colors):
        """Função usada para renderizar Polypoint2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#Polypoint2D
        # Nessa função você receberá pontos no parâmetro point, esses pontos são uma lista
        # de pontos x, y sempre na ordem. Assim point[0] é o valor da coordenada x do
        # primeiro ponto, point[1] o valor y do primeiro ponto. Já point[2] é a
        # coordenada x do segundo ponto e assim por diante. Assuma a quantidade de pontos
        # pelo tamanho da lista e assuma que sempre vira uma quantidade par de valores.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Polypoint2D
        # você pode assumir inicialmente o desenho dos pontos com a cor emissiva (emissiveColor).

        chosen_color = [int(i*255) for i in colors["emissiveColor"]]

        points = list(zip((point[::2]), point[1::2]))


        # Rever a correção para int
        for p in points:
            gpu.GPU.draw_pixel([int(p[0]), int(p[1])], gpu.GPU.RGB8, chosen_color)


    @staticmethod
    def polyline2D(lineSegments, colors):
        """Função usada para renderizar Polyline2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#Polyline2D
        # Nessa função você receberá os pontos de uma linha no parâmetro lineSegments, esses
        # pontos são uma lista de pontos x, y sempre na ordem. Assim point[0] é o valor da
        # coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto. Já point[2] é
        # a coordenada x do segundo ponto e assim por diante. Assuma a quantidade de pontos
        # pelo tamanho da lista. A quantidade mínima de pontos são 2 (4 valores), porém a
        # função pode receber mais pontos para desenhar vários segmentos. Assuma que sempre
        # vira uma quantidade par de valores.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Polyline2D
        # você pode assumir inicialmente o desenho das linhas com a cor emissiva (emissiveColor).

        points = list(zip((lineSegments[::2]), lineSegments[1::2]))
        print(points)

        chosen_color = [int(i*255) for i in colors["emissiveColor"]]


        def desenha_linha(p0, p1, color):


            dx = p1[0] - p0[0]
            dy = p1[1] - p0[1]

            steps = round(max(abs(dx), abs(dy)))

            if steps == 0:
                GL.draw([round(p0[0]), round(p0[1])], color)

            angle_x = dx/steps
            angle_y = dy/steps

            for i in range(steps+1):

                u = p0[0] + angle_x * i
                v = p0[1] + angle_y * i

                GL.draw([int(u), int(v)], color)

            return


        for i in range(len(points)):
            if not i >= len(points)-1:
                p0 = points[i]
                p1 = points[i+1]
                desenha_linha(p0,p1, chosen_color)


    # Não funciona o algoritmo de desenhar a partir de vizinhos.
    # Eventualmente implementar pelo diverencial do theta
    # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    @staticmethod
    def circle2D(radius, colors):
        """Função usada para renderizar Circle2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#Circle2D
        # Nessa função você receberá um valor de raio e deverá desenhar o contorno de
        # um círculo.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Circle2D
        # você pode assumir o desenho das linhas com a cor emissiva (emissiveColor).

        print("Circle2D : radius = {0}".format(radius)) # imprime no terminal
        print("Circle2D : colors = {0}".format(colors)) # imprime no terminal as cores

        radius = 8

        chosen_color = [int(i*255) for i in colors["emissiveColor"]]

        def distance(p0, p1):

            dx = (p1[0] - p0[0])**2
            dy = (p1[1] - p0[1])**2

            return math.sqrt(dx+dy)

        def test_neighbours(radius, current_point, previous_point):

            min_diff = math.inf
            best_option = (0,0)

            for x in [-1, 0, 1]:
                for y in [-1, 0, 1]:

                    new_point = (current_point[0] + x, current_point[1] + y)
                    rad_diff = abs( radius - distance(center, new_point))

                    if new_point != current_point and new_point != previous_point:
                        if rad_diff < min_diff:
                            min_diff = rad_diff
                            best_option = new_point

            return (best_option, current_point)


        center = (GL.width//2, GL.height//2)

        starting_point = (center[0]+radius, center[1])
        current_point = starting_point
        previous_point = None
        starting = 0

        while current_point != starting_point or starting == 0:

            starting += 1
            gpu.GPU.draw_pixel([current_point[0], current_point[1]], gpu.GPU.RGB8, chosen_color)
            current_point, previous_point = test_neighbours(radius, current_point, previous_point)

            if starting > 100:
                current_point = starting_point

        # gpu.GPU.draw_pixel([pos_x, pos_y], gpu.GPU.RGB8, [255, 0, 255])  # altera pixel (u, v, tipo, r, g, b)
        # cuidado com as cores, o X3D especifica de (0,1) e o Framebuffer de (0,255)


    @staticmethod
    def triangleSet2D(vertices, colors):
        """Função usada para renderizar TriangleSet2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#TriangleSet2D
        # Nessa função você receberá os vertices de um triângulo no parâmetro vertices,
        # esses pontos são uma lista de pontos x, y sempre na ordem. Assim point[0] é o
        # valor da coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto.
        # Já point[2] é a coordenada x do segundo ponto e assim por diante. Assuma que a
        # quantidade de pontos é sempre multiplo de 3, ou seja, 6 valores ou 12 valores, etc.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o TriangleSet2D
        # você pode assumir inicialmente o desenho das linhas com a cor emissiva (emissiveColor).

        points = np.asarray(vertices, dtype=float).reshape(-1, 2)
        emissive_color = np.clip(np.asarray(colors["emissiveColor"], dtype=float), 0, 1) * 255

        # Os triângulos 2D já estão em coordenadas de tela, então não há projeção nem profundidade.
        # Cada um é rasterizado nos buffers de supersampling, o que suaviza as bordas, e o último
        # triângulo desenhado cobre os anteriores (sem teste de profundidade).
        for i in range(0, len(points) - 2, 3):
            GL.rasterize(points[i:i + 3], np.ones(3), None,
                         lambda n, attr, attr_dx, attr_dy: emissive_color, depth_test=False)

    @staticmethod
    def quat_rotation(axis, rad):
        """
        Recebe axis : [x, y, z] e rad :  o angulo de rotação em radianos
        Retorna a matriz de rotação por quatérnios (np.array 4x4)
        """

        if rad == 0:
            return np.eye(4)

        if not any(axis):
            return np.eye(4)

        axis = np.array(axis)
        axis = axis / np.linalg.norm(axis)

        # q é o vetor coluna: [x*seno(rad/2), y*seno(rad/2), z*seno(rad/2), cos(rad/2)]
        q = [u*(np.sin(rad/2)) for u in axis]
        q += [np.cos(rad/2)]

        # qi = x*seno(rad/2)
        # qj = y*seno(rad/2)
        # qk = z*seno(rad/2)
        # qr = cos(rad/2)
        qi, qj, qk, qr = q


        # Formula da matriz de rotação em quatérnios
        return np.array([
            [ 1 - 2*(qj**2 + qk**2),      2*(qi*qj - qk*qr),     2*(qi*qk + qj*qr), 0],
            [ 2*(qi * qj + qk * qr),    1 - 2*(qi**2 + qk**2),   2*(qj*qk - qi*qr), 0],
            [ 2*(qi*qk - qj*qr),        2*(qj*qk + qi*qr),   1 - 2*(qi**2 + qj**2), 0],
            [        0,                  0,                  0,            1]
        ])

    @staticmethod
    def lookAt(pos, axis, rad):
        """
        Recebe a posição (pos) e o ângulo (axis, rad) da câmera
        retorna o inverso da rotação e a matriz de translação inversa da posição
        """

        rot = GL.quat_rotation(axis, rad)
        inv_rot = np.linalg.inv(rot)

        pos = np.array(pos)

        base = np.eye(4)
        base[:3, 3] = pos
        pos = base

        inv_pos = np.linalg.inv(pos)

        return inv_rot @ inv_pos

    @staticmethod
    def projection_matrix(near, far, fovy, aspect):

        top = near * np.tan(fovy/2)
        right = top * aspect

        proj_m = np.diag([near/right, near/top, -((far+near)/(far-near)), 0])
        proj_m[2,3] = (-2*far*near)/(far-near)
        proj_m[3,2] = -1

        return proj_m

    @staticmethod
    def homogenous_div(m):
        return m @ np.diag(1/m[3])

    @staticmethod
    def scaling(axis):
        return np.diag([*axis, 1])

    @staticmethod
    def translate(axis):
        base = np.eye(4)
        base[:3,3] = axis

        return base


    @staticmethod
    def screen_matrix(W, H):
        matrix = np.diag((W/2, -(H/2), 1, 1))
        matrix[:2,3] = np.array([W/2, H/2])

        return matrix

    @staticmethod
    def normalize(v):
        """Normaliza vetores (um por linha, ou um vetor só); os vetores nulos continuam nulos."""
        v = np.asarray(v, dtype=float)
        norm = np.linalg.norm(v, axis=-1, keepdims=True)
        return np.divide(v, norm, out=np.zeros_like(v), where=norm > 1e-12)

    @staticmethod
    def split_faces(index):
        """Divide uma lista de índices separada por -1 em uma lista de faces."""
        faces, current = [], []
        for i in index:
            if i == -1:
                if current:
                    faces.append(current)
                    current = []
            else:
                current.append(i)
        if current:
            faces.append(current)

        return faces

    @staticmethod
    def get_texture(name):
        """Lê a textura (uma única vez) e gera todos os níveis do mipmap."""
        if name not in GL.texture_cache:
            image = np.asarray(gpu.GPU.load_texture(name))

            # Garante 3 canais (RGB): tons de cinza são repetidos e o canal alfa é descartado
            if image.ndim == 2:
                image = image[:, :, None]
            if image.shape[2] < 3:
                image = np.repeat(image[:, :, :1], 3, axis=2)
            image = image[:, :, :3].astype(np.float32)

            # load_texture devolve a imagem transposta (índices [x][y]). Inverte o y para que
            # v = 0 fique embaixo, como no X3D
            levels = [np.ascontiguousarray(np.flip(image, axis=1))]

            # Cada nível seguinte tem metade do tamanho: a média de blocos de 2x2 texels
            while min(levels[-1].shape[:2]) >= 2:
                prev = levels[-1]
                w2, h2 = prev.shape[0] // 2, prev.shape[1] // 2
                levels.append(prev[:2 * w2, :2 * h2].reshape(w2, 2, h2, 2, 3).mean(axis=(1, 3)))

            GL.texture_cache[name] = levels

        return GL.texture_cache[name]

    @staticmethod
    def bound_texture():
        """Mipmap da textura da forma que está sendo desenhada (None se não tiver textura)."""
        try:
            import x3d
            name = x3d.X3D.current_texture
        except (ImportError, AttributeError):
            return None

        return GL.get_texture(name[0]) if name else None

    @staticmethod
    def sample_texture(image, u, v):
        """Amostra a imagem com filtro bilinear, com a textura se repetindo (repeatS/T = TRUE)."""
        w, h = image.shape[:2]

        # Centro do texel i fica em (i + 0.5) / tamanho
        x = np.nan_to_num(u * w - 0.5)
        y = np.nan_to_num(v * h - 0.5)
        x0 = np.floor(x)
        y0 = np.floor(y)
        fx = (x - x0)[:, None]
        fy = (y - y0)[:, None]

        x0 = x0.astype(np.int64)
        y0 = y0.astype(np.int64)
        xa, xb = x0 % w, (x0 + 1) % w
        ya, yb = y0 % h, (y0 + 1) % h

        top = image[xa, ya] * (1 - fx) + image[xb, ya] * fx
        bottom = image[xa, yb] * (1 - fx) + image[xb, yb] * fx

        return top * (1 - fy) + bottom * fy

    @staticmethod
    def sample_levels(levels, level, u, v):
        """Amostra cada coordenada (u, v) no nível do mipmap indicado em level."""
        colors = np.empty((len(u), 3))
        for lv in np.unique(level):
            mask = level == lv
            colors[mask] = GL.sample_texture(levels[lv], u[mask], v[mask])

        return colors

    @staticmethod
    def texture_color(levels, uv, uv_dx, uv_dy):
        """Cor (0 a 255) da textura em cada pixel, escolhendo o nível do mipmap pelo tamanho do pixel.

        uv, uv_dx e uv_dy são as coordenadas de textura do pixel e dos pixels vizinhos em x e y.
        """
        w0, h0 = levels[0].shape[:2]

        # Derivadas de (u, v) em relação à tela, em texels por pixel
        dudx = (uv_dx[:, 0] - uv[:, 0]) * w0
        dvdx = (uv_dx[:, 1] - uv[:, 1]) * h0
        dudy = (uv_dy[:, 0] - uv[:, 0]) * w0
        dvdy = (uv_dy[:, 1] - uv[:, 1]) * h0

        # L = quantos texels cabem em um pixel e D = log2(L) é o nível do mipmap
        L = np.maximum(np.hypot(dudx, dvdx), np.hypot(dudy, dvdy))
        with np.errstate(divide="ignore", invalid="ignore"):
            D = np.log2(L)
        last = len(levels) - 1
        D = np.clip(np.nan_to_num(D, nan=0.0, neginf=0.0, posinf=last), 0, last)

        # Interpola entre os dois níveis vizinhos de D (filtro trilinear) para não ter saltos
        lower = np.floor(D).astype(int)
        upper = np.minimum(lower + 1, last)
        frac = (D - lower)[:, None]

        return (GL.sample_levels(levels, lower, uv[:, 0], uv[:, 1]) * (1 - frac) +
                GL.sample_levels(levels, upper, uv[:, 0], uv[:, 1]) * frac)

    @staticmethod
    def illuminate(normal, position, diffuse, emissive, specular, shininess, ambient, lights):
        """Equação de iluminação do X3D (simplificada) calculada para cada pixel.

        Irgb = emissive + SUM( luz.cor * (ambient + diffuse + specular) ), com
        ambient  = luz.ambientIntensity * diffuse * material.ambientIntensity
        diffuse  = luz.intensity * diffuse * sat(N . L)
        specular = luz.intensity * specularColor * sat(N . ((L + v) / |L + v|)) ^ (shininess * 128)

        normal e position são os vetores (n, 3) das normais e das posições no mundo, diffuse é a
        cor (3,) ou (n, 3) do material. Retorna as cores (n, 3) entre 0 e 255.
        """
        N = GL.normalize(normal)
        V = GL.normalize(GL.eye_position - position)  # do ponto até o observador

        color = np.tile(np.asarray(emissive, dtype=float), (len(N), 1))
        exponent = shininess * 128

        for light in lights:
            L = light["L"]  # do ponto até a luz
            n_dot_l = N @ L

            ambient_term = light["ambient"] * ambient * diffuse
            diffuse_term = light["intensity"] * diffuse * np.clip(n_dot_l, 0, 1)[:, None]

            # Só tem brilho especular se a luz estiver do lado certo da superfície
            half = GL.normalize(V + L)
            n_dot_h = np.clip(np.sum(N * half, axis=1), 0, 1)
            specular_term = light["intensity"] * specular * (n_dot_h ** exponent * (n_dot_l > 0))[:, None]

            color += light["color"] * (ambient_term + diffuse_term + specular_term)

        return np.clip(color, 0, 1) * 255

    @staticmethod
    def interpolate(attrs, bary, inv_w):
        """Interpola os atributos dos vértices (3, k) com correção de perspectiva.

        bary são as coordenadas baricêntricas (três vetores de tamanho n) na tela e inv_w os
        valores 1/Z de cada vértice. Cada atributo é dividido por Z, interpolado na tela e
        depois multiplicado pelo Z do ponto: V = Z * (a * V0/Z0 + b * V1/Z1 + c * V2/Z2),
        com 1/Z = a/Z0 + b/Z1 + c/Z2
        """
        weights = np.stack([bary[0] * inv_w[0], bary[1] * inv_w[1], bary[2] * inv_w[2]], axis=1)
        inv_z = np.maximum(weights.sum(axis=1), 1e-12)

        return (weights @ attrs) / inv_z[:, None]

    @staticmethod
    def rasterize(scr, w, attrs, shade, alpha=0.0, depth_test=True, derivs=False):
        """Rasteriza um triângulo já projetado nos buffers de supersampling.

        scr   : (3, 2) coordenadas de tela dos vértices
        w     : (3,) distância de cada vértice até a câmera (sempre maior que zero)
        attrs : (3, k) atributos dos vértices, interpolados com correção de perspectiva (ou None)
        shade : função(n, attr, attr_dx, attr_dy) que devolve a cor (0 a 255) de cada um dos n
                pixels (matriz (n, 3), ou só uma cor (3,) se for igual para todos). attr são os
                atributos interpolados no pixel e attr_dx, attr_dy os dos pixels vizinhos em x e
                y (só calculados se derivs for True, usados pelo mipmap)
        alpha : transparência do material (0 opaco, 1 invisível)
        depth_test : se False, o triângulo cobre o que já foi desenhado (usado no 2D)

        A cobertura e a profundidade são calculadas em cada um dos 4 samples do pixel, enquanto
        a cor (textura, iluminação) é calculada uma única vez por pixel, no centroide dos
        samples cobertos, e depois copiada para os samples visíveis.
        """
        x0, y0 = scr[0]
        x1, y1 = scr[1]
        x2, y2 = scr[2]

        # O dobro da área com sinal: triângulos degenerados ou fora do mundo não são desenhados
        area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
        if not (math.isfinite(area) and abs(area) > 1e-12):
            return
        inv_area = 1.0 / area

        # Bounding box do triângulo, limitada à área do framebuffer
        x_lo = max(math.floor(min(x0, x1, x2)), 0)
        x_hi = min(math.floor(max(x0, x1, x2)), GL.width - 1)
        y_lo = max(math.floor(min(y0, y1, y2)), 0)
        y_hi = min(math.floor(max(y0, y1, y2)), GL.height - 1)
        if x_lo > x_hi or y_lo > y_hi:
            return

        # Posição de cada sample dentro da bounding box: (linhas, colunas, 4)
        sx = np.arange(x_lo, x_hi + 1)[None, :, None] + GL.offsets[:, 0]
        sy = np.arange(y_lo, y_hi + 1)[:, None, None] + GL.offsets[:, 1]

        # Coordenadas baricêntricas de cada sample (área do sub-triângulo / área total)
        b0 = ((x1 - sx) * (y2 - sy) - (y1 - sy) * (x2 - sx)) * inv_area
        b1 = ((x2 - sx) * (y0 - sy) - (y2 - sy) * (x0 - sx)) * inv_area
        b2 = ((x0 - sx) * (y1 - sy) - (y0 - sy) * (x1 - sx)) * inv_area

        # O sample está dentro do triângulo quando as três coordenadas não são negativas
        covered = (b0 >= 0) & (b1 >= 0) & (b2 >= 0)
        if not covered.any():
            return

        # Profundidade de cada sample (distância até a câmera), com correção de perspectiva
        inv_w = (1.0 / w[0], 1.0 / w[1], 1.0 / w[2])
        depth = dbuf = visible = None
        with np.errstate(divide="ignore", invalid="ignore"):
            depth = 1.0 / (b0 * inv_w[0] + b1 * inv_w[1] + b2 * inv_w[2])
            dbuf = GL.depth_buffer[y_lo:y_hi + 1, x_lo:x_hi + 1]
            if depth_test:
                # A tolerância evita que duas faces do mesmo plano, que dividem uma aresta,
                # desenhem o mesmo sample duas vezes (o que aparece como uma linha na transparência)
                visible = covered & (depth < dbuf * (1.0 - 1e-9))
            else:
                visible = covered

        # Pixels em que algum sample vai ser desenhado
        pixels = visible.any(axis=2)
        if not pixels.any():
            return
        rows, cols = np.nonzero(pixels)
        n = len(rows)

        # Coordenadas baricêntricas do centroide dos samples cobertos de cada pixel. O centroide
        # está sempre dentro do triângulo e, com o pixel todo coberto, é o centro do pixel
        cov = covered[rows, cols]
        count = cov.sum(axis=1)
        c0 = (b0[rows, cols] * cov).sum(axis=1) / count
        c1 = (b1[rows, cols] * cov).sum(axis=1) / count
        c2 = (b2[rows, cols] * cov).sum(axis=1) / count

        attr = attr_dx = attr_dy = None
        if attrs is not None:
            attr = GL.interpolate(attrs, (c0, c1, c2), inv_w)
            if derivs:
                # Atributos nos pixels vizinhos (x + 1 e y + 1), pela variação das baricêntricas
                gx = ((y1 - y2) * inv_area, (y2 - y0) * inv_area, (y0 - y1) * inv_area)
                gy = ((x2 - x1) * inv_area, (x0 - x2) * inv_area, (x1 - x0) * inv_area)
                attr_dx = GL.interpolate(attrs, (c0 + gx[0], c1 + gx[1], c2 + gx[2]), inv_w)
                attr_dy = GL.interpolate(attrs, (c0 + gy[0], c1 + gy[1], c2 + gy[2]), inv_w)

        color = np.asarray(shade(n, attr, attr_dx, attr_dy), dtype=float)
        if color.ndim == 1:
            color = np.broadcast_to(color, (n, 3))

        # Grava a cor nos samples visíveis. Com transparência ela é misturada à cor que já
        # estava no sample: cor = cor_anterior * transparência + cor_nova * (1 - transparência)
        color_buffer = GL.color_buffer[y_lo:y_hi + 1, x_lo:x_hi + 1]
        previous = color_buffer[rows, cols]
        new = color[:, None, :]
        if alpha > 0:
            new = previous * alpha + new * (1.0 - alpha)
        color_buffer[rows, cols] = np.where(visible[rows, cols][:, :, None], new, previous)

        if depth_test:
            dbuf[visible] = depth[visible]
        GL.touched[y_lo:y_hi + 1, x_lo:x_hi + 1] |= pixels

    @staticmethod
    def triangleSet(point, colors, vertexColor=None, textureCoords=None, normals=None,
                    texture=None, cull=False):
        """Função usada para renderizar TriangleSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/rendering.html#TriangleSet
        # Nessa função você receberá pontos no parâmetro point, esses pontos são uma lista
        # de pontos x, y, e z sempre na ordem. Assim point[0] é o valor da coordenada x do
        # primeiro ponto, point[1] o valor y do primeiro ponto, point[2] o valor z da
        # coordenada z do primeiro ponto. Já point[3] é a coordenada x do segundo ponto e
        # assim por diante.
        # No TriangleSet os triângulos são informados individualmente, assim os três
        # primeiros pontos definem um triângulo, os três próximos pontos definem um novo
        # triângulo, e assim por diante.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, você pode assumir
        # inicialmente, para o TriangleSet, o desenho das linhas com a cor emissiva
        # (emissiveColor), conforme implementar novos materias você deverá suportar outros
        # tipos de cores.

        # Os outros parâmetros são usados pelas geometrias que desenham seus triângulos aqui:
        # vertexColor   : cor (r, g, b) de cada vértice, na mesma ordem de point
        # textureCoords : coordenadas (u, v) de cada vértice, na mesma ordem de point
        # normals       : normal (x, y, z) de cada vértice, no espaço do objeto. Se for None cada
        #                 triângulo usa a normal da sua face (flat shading)
        # texture       : níveis do mipmap da textura (GL.get_texture)
        # cull          : descarta as faces de trás (só para sólidos fechados, como Box e Sphere)

        pts = np.asarray(point, dtype=float).reshape(-1, 3)
        pts = pts[:3 * (len(pts) // 3)]
        if len(pts) == 0:
            return

        emissive = np.asarray(colors["emissiveColor"], dtype=float)
        diffuse = np.asarray(colors["diffuseColor"], dtype=float)
        specular = np.asarray(colors["specularColor"], dtype=float)
        shininess = float(colors["shininess"])
        ambient = float(colors["ambientIntensity"])
        alpha = min(max(float(colors["transparency"]), 0.0), 1.0)
        if alpha >= 1.0:
            return  # totalmente transparente

        # Sem nó Material o X3D não usa iluminação, a cor vem só dos vértices ou da textura.
        # Como o get_colors devolve os valores padrão nesse caso, um material 100% padrão é
        # tratado como se não existisse
        lit = not (np.allclose(diffuse, 0.8) and not emissive.any() and not specular.any()
                   and abs(shininess - 0.2) < 1e-9 and alpha == 0 and abs(ambient - 0.2) < 1e-9)

        vcolors = None
        if vertexColor is not None and len(vertexColor) >= 3 * len(pts):
            vcolors = np.asarray(vertexColor, dtype=float).reshape(-1, 3)[:len(pts)]

        uvs = None
        if texture is not None and textureCoords is not None and len(textureCoords) >= 2 * len(pts):
            uvs = np.asarray(textureCoords, dtype=float).reshape(-1, 2)[:len(pts)]

        # Todos os vértices: objeto -> mundo e depois mundo -> tela (matriz do viewpoint)
        T = GL.transform_stack[-1]
        world = np.hstack([pts, np.ones((len(pts), 1))]) @ T.T
        clip = world @ GL.viewpoint_val.T
        w_all = clip[:, 3]
        with np.errstate(divide="ignore", invalid="ignore"):
            screen = clip[:, :2] / w_all[:, None]

        P3 = world[:, :3].reshape(-1, 3, 3)  # posição no mundo dos vértices de cada triângulo
        S3 = screen.reshape(-1, 3, 2)
        W3 = w_all.reshape(-1, 3)

        # Não há recorte (clipping): triângulos com algum vértice atrás do plano near são descartados
        valid = (W3 > GL.near).all(axis=1) & np.isfinite(S3).all(axis=(1, 2))

        # Normal de cada face no mundo (aponta para fora quando os vértices estão no sentido
        # anti-horário) e se a face está virada para a câmera. Uma escala negativa inverte
        # o sentido dos vértices
        if lit or cull:
            det_sign = 1.0 if np.linalg.det(T[:3, :3]) >= 0 else -1.0
            face_normal = np.cross(P3[:, 1] - P3[:, 0], P3[:, 2] - P3[:, 0]) * det_sign
            length = np.linalg.norm(face_normal, axis=1)
            facing = np.einsum("ij,ij->i", face_normal, GL.eye_position - P3[:, 0])
            valid &= length > 1e-12
            if cull:
                valid &= facing > 0

        # Atributos de cada vértice que serão interpolados: [normal, posição] [cor] [u, v]
        parts = []
        if lit:
            if normals is not None:
                # As normais são transformadas pela inversa transposta da matriz do objeto
                inverse = np.linalg.pinv(T[:3, :3])
                vertex_normal = GL.normalize(np.asarray(normals, dtype=float).reshape(-1, 3)[:len(pts)] @ inverse)
                vertex_normal = vertex_normal.reshape(-1, 3, 3)
            else:
                unit = face_normal / np.maximum(length, 1e-12)[:, None]
                vertex_normal = np.repeat(unit[:, None, :], 3, axis=1)
            # Nas faces de trás a normal é invertida, para iluminar o lado que a câmera enxerga
            vertex_normal = np.where((facing < 0)[:, None, None], -vertex_normal, vertex_normal)
            parts += [vertex_normal, P3]
            o_normal, o_position = 0, 3
        if vcolors is not None:
            o_color = sum(p.shape[2] for p in parts)
            parts.append(vcolors.reshape(-1, 3, 3))
        if uvs is not None:
            o_uv = sum(p.shape[2] for p in parts)
            parts.append(uvs.reshape(-1, 3, 2))
        attributes = np.concatenate(parts, axis=2) if parts else None

        # Luzes que iluminam a cena, o headlight sempre aponta para onde a câmera olha
        lights = list(GL.lights)
        if GL.headlight:
            lights.append({"color": np.ones(3), "intensity": 1.0, "ambient": 0.0,
                           "L": GL.eye_rotation[:, 2]})

        def shade(n, attr, attr_dx, attr_dy):
            # Cor base do pixel (a difusa do material, ou a de vértice, ou a da textura)
            base = None
            if uvs is not None:
                base = GL.texture_color(texture, attr[:, o_uv:o_uv + 2], attr_dx[:, o_uv:o_uv + 2],
                                        attr_dy[:, o_uv:o_uv + 2]) / 255.0
            elif vcolors is not None:
                base = attr[:, o_color:o_color + 3]

            if not lit:
                return np.clip(np.ones(3) if base is None else base, 0, 1) * 255

            if base is None:
                base = diffuse
            return GL.illuminate(attr[:, o_normal:o_normal + 3], attr[:, o_position:o_position + 3],
                                 base, emissive, specular, shininess, ambient, lights)

        for t in np.nonzero(valid)[0]:
            GL.rasterize(S3[t], W3[t], None if attributes is None else attributes[t], shade,
                         alpha, derivs=uvs is not None)

    @staticmethod
    def viewpoint(position, orientation, fieldOfView):
        """Função usada para renderizar (na verdade coletar os dados) de Viewpoint."""
        # Na função de viewpoint você receberá a posição, orientação e campo de visão da
        # câmera virtual. Use esses dados para poder calcular e criar a matriz de projeção
        # perspectiva para poder aplicar nos pontos dos objetos geométricos.

        # Coordenadas da câmera
        lkat = GL.lookAt(position, orientation[:3], orientation[3])

        # Projeção perspectiva
        aspect = GL.width/GL.height
        p_proj = GL.projection_matrix(GL.near, GL.far, fieldOfView, aspect)

        # Screen Transform
        screen = GL.screen_matrix(GL.width, GL.height)

        vp = screen @ p_proj @ lkat

        # Matriz de viewpoint sem divisão homogênea
        GL.viewpoint_val = vp

        # Onde a câmera está e para onde ela aponta, usados no cálculo da iluminação
        GL.eye_position = np.array(position, dtype=float)
        GL.eye_rotation = GL.quat_rotation(orientation[:3], orientation[3])[:3, :3]

    @staticmethod
    def transform_in(translation, scale, rotation):
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""
        # A função transform_in será chamada quando se entrar em um nó X3D do tipo Transform
        # do grafo de cena. Os valores passados são a escala em um vetor [x, y, z]
        # indicando a escala em cada direção, a translação [x, y, z] nas respectivas
        # coordenadas e finalmente a rotação por [x, y, z, t] sendo definida pela rotação
        # do objeto ao redor do eixo x, y, z por t radianos, seguindo a regra da mão direita.
        # ESSES NÃO SÃO OS VALORES DE QUATÉRNIOS AS CONTAS AINDA PRECISAM SER FEITAS.
        # Quando se entrar em um nó transform se deverá salvar a matriz de transformação dos
        # modelos do mundo para depois potencialmente usar em outras chamadas.
        # Quando começar a usar Transforms dentre de outros Transforms, mais a frente no curso
        # Você precisará usar alguma estrutura de dados pilha para organizar as matrizes.

        # Os valores podem vir de uma animação (ROUTE), então não dá para testá-los com "if valor"
        T = np.eye(4)

        if scale is not None and len(scale) == 3:
            T = GL.scaling(scale) @ T
        if rotation is not None and len(rotation) == 4:
            T = GL.quat_rotation(rotation[:3], rotation[3]) @ T
        if translation is not None and len(translation) == 3:
            T = GL.translate(translation) @ T

        GL.transform_stack.append( GL.transform_stack[-1] @ T)

    @staticmethod
    def transform_out():
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""
        # A função transform_out será chamada quando se sair em um nó X3D do tipo Transform do
        # grafo de cena. Não são passados valores, porém quando se sai de um nó transform se
        # deverá recuperar a matriz de transformação dos modelos do mundo da estrutura de
        # pilha implementada.

        GL.transform_stack.pop(-1)

    @staticmethod
    def draw_strips(point, strips, colors):
        """Desenha tiras de triângulos, dadas como listas com os índices dos vértices."""
        pts = np.asarray(point, dtype=float).reshape(-1, 3)

        # Cada tira vira os triângulos (i, i+1, i+2). Nos de índice ímpar os dois primeiros
        # vértices são trocados para que todos fiquem no mesmo sentido (anti-horário)
        triangles = []
        for strip in strips:
            for i in range(len(strip) - 2):
                a, b, c = strip[i], strip[i + 1], strip[i + 2]
                triangles.append((b, a, c) if i % 2 else (a, b, c))
        if not triangles:
            return
        tri = np.array(triangles)

        # Normal de cada vértice: média das normais das faces que compartilham o vértice
        face_normal = GL.normalize(np.cross(pts[tri[:, 1]] - pts[tri[:, 0]],
                                            pts[tri[:, 2]] - pts[tri[:, 0]]))
        vertex_normal = np.zeros_like(pts)
        for corner in range(3):
            np.add.at(vertex_normal, tri[:, corner], face_normal)
        vertex_normal = GL.normalize(vertex_normal)

        GL.triangleSet(pts[tri].reshape(-1), colors, normals=vertex_normal[tri].reshape(-1))

    @staticmethod
    def triangleStripSet(point, stripCount, colors):
        """Função usada para renderizar TriangleStripSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/rendering.html#TriangleStripSet
        # A função triangleStripSet é usada para desenhar tiras de triângulos interconectados,
        # você receberá as coordenadas dos pontos no parâmetro point, esses pontos são uma
        # lista de pontos x, y, e z sempre na ordem. Assim point[0] é o valor da coordenada x
        # do primeiro ponto, point[1] o valor y do primeiro ponto, point[2] o valor z da
        # coordenada z do primeiro ponto. Já point[3] é a coordenada x do segundo ponto e assim
        # por diante. No TriangleStripSet a quantidade de vértices a serem usados é informado
        # em uma lista chamada stripCount (perceba que é uma lista). Ligue os vértices na ordem,
        # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
        # depois 2, 3 e 4, e assim por diante. Cuidado com a orientação dos vértices, ou seja,
        # todos no sentido horário ou todos no sentido anti-horário, conforme especificado.

        # Cada tira usa stripCount[i] vértices seguidos, começando onde a anterior terminou
        strips, start = [], 0
        for count in stripCount:
            strips.append(list(range(start, start + count)))
            start += count

        GL.draw_strips(point, strips, colors)

    @staticmethod
    def indexedTriangleStripSet(point, index, colors):
        """Função usada para renderizar IndexedTriangleStripSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/rendering.html#IndexedTriangleStripSet
        # A função indexedTriangleStripSet é usada para desenhar tiras de triângulos
        # interconectados, você receberá as coordenadas dos pontos no parâmetro point, esses
        # pontos são uma lista de pontos x, y, e z sempre na ordem. Assim point[0] é o valor
        # da coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto, point[2]
        # o valor z da coordenada z do primeiro ponto. Já point[3] é a coordenada x do
        # segundo ponto e assim por diante. No IndexedTriangleStripSet uma lista informando
        # como conectar os vértices é informada em index, o valor -1 indica que a lista
        # acabou. A ordem de conexão será de 3 em 3 pulando um índice. Por exemplo: o
        # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
        # depois 2, 3 e 4, e assim por diante. Cuidado com a orientação dos vértices, ou seja,
        # todos no sentido horário ou todos no sentido anti-horário, conforme especificado.

        GL.draw_strips(point, GL.split_faces(index), colors)

    @staticmethod
    def indexedFaceSet(coord, coordIndex, colorPerVertex, color, colorIndex,
                       texCoord, texCoordIndex, colors, current_texture):
        """Função usada para renderizar IndexedFaceSet."""

        # Parte 1

        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#IndexedFaceSet
        # A função indexedFaceSet é usada para desenhar malhas de triângulos. Ela funciona de
        # forma muito simular a IndexedTriangleStripSet porém com mais recursos.
        # Você receberá as coordenadas dos pontos no parâmetro cord, esses
        # pontos são uma lista de pontos x, y, e z sempre na ordem. Assim coord[0] é o valor
        # da coordenada x do primeiro ponto, coord[1] o valor y do primeiro ponto, coord[2]
        # o valor z da coordenada z do primeiro ponto. Já coord[3] é a coordenada x do
        # segundo ponto e assim por diante. No IndexedFaceSet uma lista de vértices é informada
        # em coordIndex, o valor -1 indica que a lista acabou.
        # A ordem de conexão não possui uma ordem oficial, mas em geral se o primeiro ponto com os dois
        # seguintes e depois este mesmo primeiro ponto com o terçeiro e quarto ponto. Por exemplo: numa
        # sequencia 0, 1, 2, 3, 4, -1 o primeiro triângulo será com os vértices 0, 1 e 2, depois serão
        # os vértices 0, 2 e 3, e depois 0, 3 e 4, e assim por diante, até chegar no final da lista.


        # Parte 2


        # Adicionalmente essa implementação do IndexedFace aceita cores por vértices, assim
        # se a flag colorPerVertex estiver habilitada, os vértices também possuirão cores
        # que servem para definir a cor interna dos poligonos, para isso faça um cálculo
        # baricêntrico de que cor deverá ter aquela posição. Da mesma forma se pode definir uma
        # textura para o poligono, para isso, use as coordenadas de textura e depois aplique a
        # cor da textura conforme a posição do mapeamento. Dentro da classe GPU já está
        # implementadado um método para a leitura de imagens.

        if not coord or not coordIndex:
            return

        points = np.asarray(coord, dtype=float).reshape(-1, 3)
        faces = GL.split_faces(coordIndex)

        # Cores: por vértice de cada face (colorIndex, ou coordIndex se não houver) ou uma por face
        palette = np.asarray(color, dtype=float).reshape(-1, 3) if color else None
        color_faces, face_colors = faces, None
        if palette is not None:
            if colorPerVertex and colorIndex:
                color_faces = GL.split_faces(colorIndex)
            elif not colorPerVertex:
                face_colors = [c for c in colorIndex if c >= 0] if colorIndex else list(range(len(faces)))

        # Textura: só é usada se a forma tem textura e coordenadas de textura
        texture, uv_table, uv_faces = None, None, faces
        if current_texture and texCoord:
            texture = GL.get_texture(current_texture[0])
            uv_table = np.asarray(texCoord, dtype=float).reshape(-1, 2)
            if texCoordIndex:
                uv_faces = GL.split_faces(texCoordIndex)

        # Cada face vira um leque de triângulos: (0, 1, 2), (0, 2, 3), (0, 3, 4), ...
        point_index, color_index, uv_index = [], [], []
        for f, face in enumerate(faces):
            if len(face) < 3:
                continue

            vertex_colors = color_faces[f] if f < len(color_faces) else face
            if len(vertex_colors) != len(face):
                vertex_colors = face
            vertex_uvs = uv_faces[f] if f < len(uv_faces) else face
            if len(vertex_uvs) != len(face):
                vertex_uvs = face

            for k in range(1, len(face) - 1):
                for corner in (0, k, k + 1):
                    point_index.append(face[corner])
                    if palette is not None:
                        if face_colors is not None:
                            color_index.append(face_colors[min(f, len(face_colors) - 1)])
                        else:
                            color_index.append(vertex_colors[corner])
                    if uv_table is not None:
                        uv_index.append(vertex_uvs[corner])

        if not point_index:
            return

        GL.triangleSet(points[point_index].reshape(-1), colors,
                       palette[color_index].reshape(-1) if palette is not None else None,
                       uv_table[uv_index].reshape(-1) if uv_table is not None else None,
                       texture=texture)

    @staticmethod
    def box(size, colors):
        """Função usada para renderizar Boxes."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Box
        # A função box é usada para desenhar paralelepípedos na cena. O Box é centrada no
        # (0, 0, 0) no sistema de coordenadas local e alinhado com os eixos de coordenadas
        # locais. O argumento size especifica as extensões da caixa ao longo dos eixos X, Y
        # e Z, respectivamente, e cada valor do tamanho deve ser maior que zero. Para desenha
        # essa caixa você vai provavelmente querer tesselar ela em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        hx, hy, hz = size[0] / 2, size[1] / 2, size[2] / 2

        # Cada face tem 4 vértices no sentido anti-horário visto de fora, começando pelo canto
        # inferior esquerdo (é nesse canto que a textura começa)
        faces = [
            [(-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)],       # frente (+z)
            [(hx, -hy, -hz), (-hx, -hy, -hz), (-hx, hy, -hz), (hx, hy, -hz)],   # trás (-z)
            [(hx, -hy, hz), (hx, -hy, -hz), (hx, hy, -hz), (hx, hy, hz)],       # direita (+x)
            [(-hx, -hy, -hz), (-hx, -hy, hz), (-hx, hy, hz), (-hx, hy, -hz)],   # esquerda (-x)
            [(-hx, hy, hz), (hx, hy, hz), (hx, hy, -hz), (-hx, hy, -hz)],       # topo (+y)
            [(-hx, -hy, -hz), (hx, -hy, -hz), (hx, -hy, hz), (-hx, -hy, hz)],   # base (-y)
        ]
        corner_uv = [(0, 0), (1, 0), (1, 1), (0, 1)]

        coords, uvs = [], []
        for quad in faces:
            for a, b, c in ((0, 1, 2), (0, 2, 3)):
                coords += [quad[a], quad[b], quad[c]]
                uvs += [corner_uv[a], corner_uv[b], corner_uv[c]]

        # Se a caixa tem textura, cada face recebe a imagem inteira
        texture = GL.bound_texture()
        GL.triangleSet(np.array(coords).reshape(-1), colors,
                       textureCoords=np.array(uvs).reshape(-1) if texture is not None else None,
                       texture=texture, cull=True)

    @staticmethod
    def sphere(radius, colors):
        """Função usada para renderizar Esferas."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Sphere
        # A função sphere é usada para desenhar esferas na cena. O esfera é centrada no
        # (0, 0, 0) no sistema de coordenadas local. O argumento radius especifica o
        # raio da esfera que está sendo criada. Para desenha essa esfera você vai
        # precisar tesselar ela em triângulos, para isso encontre os vértices e defina
        # os triângulos.

        stacks, slices = 16, 32

        # Grade de pontos da esfera: theta vai do polo norte (+y) ao sul e phi dá a volta no eixo y
        theta, phi = np.meshgrid(np.linspace(0, math.pi, stacks + 1),
                                 np.linspace(0, 2 * math.pi, slices + 1), indexing="ij")
        unit = np.stack([np.sin(theta) * np.cos(phi), np.cos(theta), np.sin(theta) * np.sin(phi)], axis=-1)
        points = radius * unit

        # Cada célula da grade (A, B, C, D) vira dois triângulos anti-horários vistos de fora:
        # (A, D, C) e (A, C, B). A normal da esfera em cada ponto é o próprio vetor unitário
        i, j = np.meshgrid(np.arange(stacks), np.arange(slices), indexing="ij")
        i, j = i.ravel(), j.ravel()
        corners = [((i, j), (i, j + 1), (i + 1, j + 1)), ((i, j), (i + 1, j + 1), (i + 1, j))]

        coords = np.concatenate([np.stack([points[c] for c in tri], axis=1) for tri in corners])
        normals = np.concatenate([np.stack([unit[c] for c in tri], axis=1) for tri in corners])

        GL.triangleSet(coords.reshape(-1), colors, normals=normals.reshape(-1), cull=True)

    @staticmethod
    def cone(bottomRadius, height, colors):
        """Função usada para renderizar Cones."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Cone
        # A função cone é usada para desenhar cones na cena. O cone é centrado no
        # (0, 0, 0) no sistema de coordenadas local. O argumento bottomRadius especifica o
        # raio da base do cone e o argumento height especifica a altura do cone.
        # O cone é alinhado com o eixo Y local. O cone é fechado por padrão na base.
        # Para desenha esse cone você vai precisar tesselar ele em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        slices = 24
        r, h = bottomRadius, height

        phi = np.linspace(0, 2 * math.pi, slices + 1)
        ring = np.stack([r * np.cos(phi), np.full_like(phi, -h / 2), r * np.sin(phi)], axis=1)
        apex = np.array([0.0, h / 2, 0.0])
        center = np.array([0.0, -h / 2, 0.0])
        slant = math.hypot(r, h)

        def side_normal(angle):
            # Normal da lateral do cone, que aponta para fora e um pouco para cima
            return np.array([h * math.cos(angle), r, h * math.sin(angle)]) / slant

        down = np.array([0.0, -1.0, 0.0])
        coords, normals = [], []
        for k in range(slices):
            middle = (phi[k] + phi[k + 1]) / 2

            # Lateral: o ápice usa a normal do meio da fatia, pois ali ela muda em cada fatia
            coords.append([apex, ring[k + 1], ring[k]])
            normals.append([side_normal(middle), side_normal(phi[k + 1]), side_normal(phi[k])])

            # Base
            coords.append([center, ring[k], ring[k + 1]])
            normals.append([down, down, down])

        GL.triangleSet(np.array(coords).reshape(-1), colors,
                       normals=np.array(normals).reshape(-1), cull=True)

    @staticmethod
    def cylinder(radius, height, colors):
        """Função usada para renderizar Cilindros."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Cylinder
        # A função cylinder é usada para desenhar cilindros na cena. O cilindro é centrado no
        # (0, 0, 0) no sistema de coordenadas local. O argumento radius especifica o
        # raio da base do cilindro e o argumento height especifica a altura do cilindro.
        # O cilindro é alinhado com o eixo Y local. O cilindro é fechado por padrão em ambas as extremidades.
        # Para desenha esse cilindro você vai precisar tesselar ele em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        slices = 24
        r, h = radius, height

        phi = np.linspace(0, 2 * math.pi, slices + 1)
        bottom = np.stack([r * np.cos(phi), np.full_like(phi, -h / 2), r * np.sin(phi)], axis=1)
        top = np.stack([r * np.cos(phi), np.full_like(phi, h / 2), r * np.sin(phi)], axis=1)
        side = np.stack([np.cos(phi), np.zeros_like(phi), np.sin(phi)], axis=1)  # normais da lateral
        top_center = np.array([0.0, h / 2, 0.0])
        bottom_center = np.array([0.0, -h / 2, 0.0])
        up = np.array([0.0, 1.0, 0.0])

        coords, normals = [], []
        for k in range(slices):
            # Lateral: dois triângulos por fatia
            coords.append([bottom[k], top[k], top[k + 1]])
            normals.append([side[k], side[k], side[k + 1]])
            coords.append([bottom[k], top[k + 1], bottom[k + 1]])
            normals.append([side[k], side[k + 1], side[k + 1]])

            # Tampas
            coords.append([top_center, top[k + 1], top[k]])
            normals.append([up, up, up])
            coords.append([bottom_center, bottom[k], bottom[k + 1]])
            normals.append([-up, -up, -up])

        GL.triangleSet(np.array(coords).reshape(-1), colors,
                       normals=np.array(normals).reshape(-1), cull=True)

    @staticmethod
    def navigationInfo(headlight):
        """Características físicas do avatar do visualizador e do modelo de visualização."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/navigation.html#NavigationInfo
        # O campo do headlight especifica se um navegador deve acender um luz direcional que
        # sempre aponta na direção que o usuário está olhando. Definir este campo como TRUE
        # faz com que o visualizador forneça sempre uma luz do ponto de vista do usuário.
        # A luz headlight deve ser direcional, ter intensidade = 1, cor = (1 1 1),
        # ambientIntensity = 0,0 e direção = (0 0 −1).

        # A direção (0 0 -1) é no espaço da câmera, então ela é calculada no triangleSet
        # a partir da rotação do Viewpoint
        GL.headlight = bool(headlight)

    @staticmethod
    def directionalLight(ambientIntensity, color, intensity, direction):
        """Luz direcional ou paralela."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/lighting.html#DirectionalLight
        # Define uma fonte de luz direcional que ilumina ao longo de raios paralelos
        # em um determinado vetor tridimensional. Possui os campos básicos ambientIntensity,
        # cor, intensidade. O campo de direção especifica o vetor de direção da iluminação
        # que emana da fonte de luz no sistema de coordenadas local. A luz é emitida ao
        # longo de raios paralelos de uma distância infinita.

        direction = np.asarray(direction, dtype=float)
        length = np.linalg.norm(direction)
        if length < 1e-12:
            return

        # L é o vetor que sai da superfície em direção à luz, ou seja, o oposto da direção da luz
        GL.lights.append({
            "ambient": float(ambientIntensity),
            "color": np.asarray(color, dtype=float),
            "intensity": float(intensity),
            "L": -direction / length,
        })

    @staticmethod
    def pointLight(ambientIntensity, color, intensity, location):
        """Luz pontual."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/lighting.html#PointLight
        # Fonte de luz pontual em um local 3D no sistema de coordenadas local. Uma fonte
        # de luz pontual emite luz igualmente em todas as direções; ou seja, é omnidirecional.
        # Possui os campos básicos ambientIntensity, cor, intensidade. Um nó PointLight ilumina
        # a geometria em um raio de sua localização. O campo do raio deve ser maior ou igual a
        # zero. A iluminação do nó PointLight diminui com a distância especificada.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("PointLight : ambientIntensity = {0}".format(ambientIntensity))
        print("PointLight : color = {0}".format(color)) # imprime no terminal
        print("PointLight : intensity = {0}".format(intensity)) # imprime no terminal
        print("PointLight : location = {0}".format(location)) # imprime no terminal

    @staticmethod
    def fog(visibilityRange, color):
        """Névoa."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/environmentalEffects.html#Fog
        # O nó Fog fornece uma maneira de simular efeitos atmosféricos combinando objetos
        # com a cor especificada pelo campo de cores com base nas distâncias dos
        # vários objetos ao visualizador. A visibilidadeRange especifica a distância no
        # sistema de coordenadas local na qual os objetos são totalmente obscurecidos
        # pela névoa. Os objetos localizados fora de visibilityRange do visualizador são
        # desenhados com uma cor de cor constante. Objetos muito próximos do visualizador
        # são muito pouco misturados com a cor do nevoeiro.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Fog : color = {0}".format(color)) # imprime no terminal
        print("Fog : visibilityRange = {0}".format(visibilityRange))

    @staticmethod
    def timeSensor(cycleInterval, loop):
        """Gera eventos conforme o tempo passa."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/time.html#TimeSensor
        # Os nós TimeSensor podem ser usados para muitas finalidades, incluindo:
        # Condução de simulações e animações contínuas; Controlar atividades periódicas;
        # iniciar eventos de ocorrência única, como um despertador;
        # Se, no final de um ciclo, o valor do loop for FALSE, a execução é encerrada.
        # Por outro lado, se o loop for TRUE no final de um ciclo, um nó dependente do
        # tempo continua a execução no próximo ciclo. O ciclo de um nó TimeSensor dura
        # cycleInterval segundos. O valor de cycleInterval deve ser maior que zero.

        # Deve retornar a fração de tempo passada em fraction_changed

        # O tempo conta a partir da primeira chamada, ou seja, a animação começa no primeiro frame
        now = time.time()  # time in seconds since the epoch as a floating point number.
        if GL.start_time is None:
            GL.start_time = now
        elapsed = now - GL.start_time

        if cycleInterval <= 0:
            return 0.0

        if loop:
            fraction_changed = (elapsed % cycleInterval) / cycleInterval
        else:
            fraction_changed = min(elapsed / cycleInterval, 1.0)  # termina no fim do ciclo

        return fraction_changed

    @staticmethod
    def find_interval(key, fraction):
        """Índice i do intervalo da animação (key[i] <= fraction < key[i+1]) e a posição s (0 a 1) nele."""
        i = int(np.searchsorted(key, fraction, side="right")) - 1
        i = min(max(i, 0), len(key) - 2)

        span = key[i + 1] - key[i]
        s = (fraction - key[i]) / span if span > 0 else 0.0

        return i, s

    @staticmethod
    def splinePositionInterpolator(set_fraction, key, keyValue, closed):
        """Interpola não linearmente entre uma lista de vetores 3D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/interpolators.html#SplinePositionInterpolator
        # Interpola não linearmente entre uma lista de vetores 3D. O campo keyValue possui
        # uma lista com os valores a serem interpolados, key possui uma lista respectiva de chaves
        # dos valores em keyValue, a fração a ser interpolada vem de set_fraction que varia de
        # zeroa a um. O campo keyValue deve conter exatamente tantos vetores 3D quanto os
        # quadros-chave no key. O campo closed especifica se o interpolador deve tratar a malha
        # como fechada, com uma transições da última chave para a primeira chave. Se os keyValues
        # na primeira e na última chave não forem idênticos, o campo closed será ignorado.

        if not key or not keyValue:
            return [0.0, 0.0, 0.0]

        values = np.asarray(keyValue, dtype=float).reshape(-1, 3)
        keys = np.asarray(key, dtype=float)[:len(values)]
        values = values[:len(keys)]
        n = len(keys)

        # Fora do intervalo das chaves vale o primeiro ou o último valor
        if n == 1 or set_fraction <= keys[0]:
            return values[0].tolist()
        if set_fraction >= keys[-1]:
            return values[-1].tolist()

        i, s = GL.find_interval(keys, set_fraction)

        # Tangentes de Catmull-Rom: T_i = (v_(i+1) - v_(i-1)) / 2. Nas pontas elas são nulas, a
        # menos que a curva seja fechada (primeiro e último valor idênticos), quando as duas
        # pontas usam a mesma tangente e a curva fica suave na emenda
        is_loop = bool(closed) and n > 2 and np.allclose(values[0], values[-1])

        def tangent(k):
            if k == 0 or k == n - 1:
                return (values[1] - values[n - 2]) / 2 if is_loop else np.zeros(3)
            return (values[k + 1] - values[k - 1]) / 2

        # Interpolação de Hermite: v = S^T H C, com S = [s^3, s^2, s, 1] e C = [v_i, v_(i+1), T_i, T_(i+1)]
        S = np.array([s ** 3, s ** 2, s, 1.0])
        H = np.array([[ 2, -2,  1,  1],
                      [-3,  3, -2, -1],
                      [ 0,  0,  1,  0],
                      [ 1,  0,  0,  0]])
        C = np.array([values[i], values[i + 1], tangent(i), tangent(i + 1)])

        value_changed = (S @ H @ C).tolist()

        return value_changed

    @staticmethod
    def quaternion(rotation):
        """Quatérnio unitário [x, y, z, w] de uma rotação [x, y, z, ângulo]."""
        axis = GL.normalize(np.asarray(rotation[:3], dtype=float))
        if not axis.any():
            return np.array([0.0, 0.0, 0.0, 1.0])

        return np.append(axis * math.sin(rotation[3] / 2), math.cos(rotation[3] / 2))

    @staticmethod
    def slerp(q0, q1, s):
        """Interpolação esférica entre dois quatérnios unitários, pelo caminho mais curto."""
        dot = float(np.dot(q0, q1))
        if dot < 0:  # os dois sinais representam a mesma rotação, fica com o mais próximo
            q1, dot = -q1, -dot

        if dot > 0.9995:  # quase iguais: a interpolação linear já é suficiente
            q = q0 + s * (q1 - q0)
        else:
            theta = math.acos(dot)
            q = (math.sin((1 - s) * theta) * q0 + math.sin(s * theta) * q1) / math.sin(theta)

        return q / np.linalg.norm(q)

    @staticmethod
    def axis_angle(q):
        """Rotação [x, y, z, ângulo] de um quatérnio unitário [x, y, z, w]."""
        if q[3] < 0:
            q = -q
        w = min(max(float(q[3]), -1.0), 1.0)

        sin_half = math.sqrt(max(1 - w * w, 0.0))
        if sin_half < 1e-9:
            return [0.0, 0.0, 1.0, 0.0]

        return [float(q[0] / sin_half), float(q[1] / sin_half), float(q[2] / sin_half),
                2 * math.acos(w)]

    @staticmethod
    def orientationInterpolator(set_fraction, key, keyValue):
        """Interpola entre uma lista de valores de rotação especificos."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/interpolators.html#OrientationInterpolator
        # Interpola rotações são absolutas no espaço do objeto e, portanto, não são cumulativas.
        # Uma orientação representa a posição final de um objeto após a aplicação de uma rotação.
        # Um OrientationInterpolator interpola entre duas orientações calculando o caminho mais
        # curto na esfera unitária entre as duas orientações. A interpolação é linear em
        # comprimento de arco ao longo deste caminho. Os resultados são indefinidos se as duas
        # orientações forem diagonalmente opostas. O campo keyValue possui uma lista com os
        # valores a serem interpolados, key possui uma lista respectiva de chaves
        # dos valores em keyValue, a fração a ser interpolada vem de set_fraction que varia de
        # zeroa a um. O campo keyValue deve conter exatamente tantas rotações 3D quanto os
        # quadros-chave no key.

        if not key or not keyValue:
            return [0.0, 0.0, 1.0, 0.0]

        values = np.asarray(keyValue, dtype=float).reshape(-1, 4)
        keys = np.asarray(key, dtype=float)[:len(values)]
        values = values[:len(keys)]

        # Fora do intervalo das chaves vale o primeiro ou o último valor
        if len(keys) == 1 or set_fraction <= keys[0]:
            return values[0].tolist()
        if set_fraction >= keys[-1]:
            return values[-1].tolist()

        i, s = GL.find_interval(keys, set_fraction)

        # SLERP: interpola os quatérnios das duas rotações e volta para eixo e ângulo
        q = GL.slerp(GL.quaternion(values[i]), GL.quaternion(values[i + 1]), s)

        value_changed = GL.axis_angle(q)

        return value_changed

    # Para o futuro (Não para versão atual do projeto.)
    def vertex_shader(self, shader):
        """Para no futuro implementar um vertex shader."""

    def fragment_shader(self, shader):
        """Para no futuro implementar um fragment shader."""
