import pygame as pg
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import pyrr


class Cube:

    def __init__(self, position, eulers):

        self.position = np.array(position, dtype=np.float32)
        self.eulers = np.array(eulers, dtype=np.float32)


class App:
    # Model OBJ, texture IMAGE FORMAT ,  position [x,y,z], eulers/rotation [x,y,z]
    def __init__(self, model: str, texture: str, position, eulers):
        # initialise pygame
        pg.init()
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK,
                                    pg.GL_CONTEXT_PROFILE_CORE)
        pg.display.set_mode((1080, 720), pg.OPENGL | pg.DOUBLEBUF)
        self.clock = pg.time.Clock()
        # initialise opengl
        glClearColor(0.1, 0.2, 0.2, 1)
        self.cube_mesh = Mesh(model)
        self.shader = self.createShader(
            "Shaders/vertex.txt", "Shaders/fragment.txt")
        glUseProgram(self.shader)
        glUniform1i(glGetUniformLocation(self.shader, "imageTexture"), 0)
        glEnable(GL_DEPTH_TEST)

        self.wood_texture = Material(texture)

        self.cube = Cube(position, eulers)

        projection_transform = pyrr.matrix44.create_perspective_projection(
            fovy=45, aspect=1080/720,
            near=0.1, far=10, dtype=np.float32
        )
        glUniformMatrix4fv(
            glGetUniformLocation(self.shader, "projection"),
            1, GL_FALSE, projection_transform
        )
        self.modelMatrixLocation = glGetUniformLocation(self.shader, "model")
        self.mainLoop()

    def createShader(self, vertexFilepath, fragmentFilepath):

        with open(vertexFilepath, 'r') as f:
            vertex_src = f.readlines()

        with open(fragmentFilepath, 'r') as f:
            fragment_src = f.readlines()

        shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER),
                                compileShader(fragment_src, GL_FRAGMENT_SHADER))

        return shader

    def mainLoop(self):

        # arrow variables for key bindings
        arrow_down = False
        arrow_Up = False
        arrow_left = False
        arrow_right = False

        running = True
        while (running):
            # check events
            for event in pg.event.get():
                # Key down for movement
                if(event.type == pg.KEYDOWN):
                    if event.key == pg.K_DOWN:
                        arrow_down = True

                    if event.key == pg.K_UP:
                        arrow_Up = True

                    if event.key == pg.K_LEFT:
                        arrow_left = True

                    if event.key == pg.K_RIGHT:
                        arrow_right = True
                # key up for stoping movement
                elif(event.type == pg.KEYUP):
                    if event.key == pg.K_DOWN:
                        arrow_down = False

                    if event.key == pg.K_UP:
                        arrow_Up = False

                    if event.key == pg.K_LEFT:
                        arrow_left = False

                    if event.key == pg.K_RIGHT:
                        arrow_right = False
                if (event.type == pg.QUIT):
                    running = False

            if(arrow_down):
                self.cube.eulers[0] -= 1
            if(arrow_Up):
                self.cube.eulers[0] += 1
            if(arrow_left):
                self.cube.eulers[2] += 1
            if(arrow_right):
                self.cube.eulers[2] -= 1

            # reseting variables for memory efficiency
            if self.cube.eulers[0] > 360 or self.cube.eulers[0] < -360:
                self.cube.eulers[0] = 0
            elif self.cube.eulers[2] > 360 or self.cube.eulers[2] < -360:
                self.cube.eulers[2] = 0

            # update cube
            # self.cube.eulers[2] += 0.25
            # if self.cube.eulers[2] > 360:
            #     self.cube.eulers[2] -= 360

            # refresh screen
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glUseProgram(self.shader)

            model_transform = pyrr.matrix44.create_identity(dtype=np.float32)

            model_transform = pyrr.matrix44.multiply(
                m1=model_transform,
                m2=pyrr.matrix44.create_from_eulers(
                    eulers=np.radians(self.cube.eulers), dtype=np.float32
                )
            )
            model_transform = pyrr.matrix44.multiply(
                m1=model_transform,
                m2=pyrr.matrix44.create_from_translation(
                    vec=np.array(self.cube.position), dtype=np.float32
                )
            )
            glUniformMatrix4fv(self.modelMatrixLocation, 1,
                               GL_FALSE, model_transform)
            self.wood_texture.use()
            glBindVertexArray(self.cube_mesh.vao)
            glDrawArrays(GL_TRIANGLES, 0, self.cube_mesh.vertex_count)

            pg.display.flip()

            # timing
            self.clock.tick(60)
        self.quit()

    def quit(self):
        self.cube_mesh.destroy()
        self.wood_texture.destroy()
        glDeleteProgram(self.shader)
        pg.quit()


class Mesh:
    def __init__(self, filename):
        # x, y, z, s, t, nx, ny, nz
        self.vertices = self.loadMesh(filename)
        self.vertex_count = len(self.vertices)//8
        self.vertices = np.array(self.vertices, dtype=np.float32)

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes,
                     self.vertices, GL_STATIC_DRAW)
        # position
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))
        # texture
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE,
                              32, ctypes.c_void_p(12))

    def loadMesh(self, filename):

        # raw, unassembled data
        v = []
        vt = []
        vn = []

        # final, assembled and packed result
        vertices = []

        # open the obj file and read the data
        with open(filename, 'r') as f:
            line = f.readline()
            while line:
                firstSpace = line.find(" ")
                flag = line[0:firstSpace]
                if flag == "v":
                    # vertex
                    line = line.replace("v ", "")
                    line = line.split(" ")
                    l = [float(x) for x in line]
                    v.append(l)
                elif flag == "vt":
                    # texture coordinate
                    line = line.replace("vt ", "")
                    line = line.split(" ")
                    l = [float(x) for x in line]
                    vt.append(l)
                elif flag == "vn":
                    # normal
                    line = line.replace("vn ", "")
                    line = line.split(" ")
                    l = [float(x) for x in line]
                    vn.append(l)
                elif flag == "f":
                    # face, three or more vertices in v/vt/vn form
                    line = line.replace("f ", "")
                    line = line.replace("\n", "")
                    # get the individual vertices for each line
                    line = line.split(" ")
                    faceVertices = []
                    faceTextures = []
                    faceNormals = []
                    for vertex in line:
                        # break out into [v,vt,vn],
                        # correct for 0 based indexing.
                        l = vertex.split("/")
                        position = int(l[0]) - 1
                        faceVertices.append(v[position])
                        texture = int(l[1]) - 1
                        faceTextures.append(vt[texture])
                        normal = int(l[2]) - 1
                        faceNormals.append(vn[normal])
                    # obj file uses triangle fan format for each face individually.
                    # unpack each face
                    triangles_in_face = len(line) - 2

                    vertex_order = []
                    """
                        eg. 0,1,2,3 unpacks to vertices: [0,1,2,0,2,3]
                    """
                    for i in range(triangles_in_face):
                        vertex_order.append(0)
                        vertex_order.append(i+1)
                        vertex_order.append(i+2)
                    for i in vertex_order:
                        for x in faceVertices[i]:
                            vertices.append(x)
                        for x in faceTextures[i]:
                            vertices.append(x)
                        for x in faceNormals[i]:
                            vertices.append(x)
                line = f.readline()
        return vertices

    def destroy(self):
        glDeleteVertexArrays(1, (self.vao,))
        glDeleteBuffers(1, (self.vbo,))


class Material:

    def __init__(self, filepath):
        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        image = pg.image.load(filepath).convert()
        image_width, image_height = image.get_rect().size
        img_data = pg.image.tostring(image, 'RGBA')
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image_width,
                     image_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
        glGenerateMipmap(GL_TEXTURE_2D)

    def use(self):
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture)

    def destroy(self):
        glDeleteTextures(1, (self.texture,))


position = [0, -0.3, -3]
eulers = [0, 0, 0]

model = ""
texture = ""

print("Modelos: \n[1] TIGER_TANK \n[2] Mask \n[3] Eyeball")
model_option = int(input("Seleccione el modelo que desea cargar: \n"))
if model_option == 1:
    model = "models\Tiger2.obj"
    texture = "models\Texture\Tiger.png"

if model_option == 2:
    model = "models\mask.obj"
    texture = "models\Texture\Steve.png"
    position = [0, -1, -4]

if model_option == 3:
    model = "models\eyeball.obj"
    texture = "models\Texture\Eye_D.jpg"
    position = [0, -0.5, -10]
myApp = App(model, texture, position, eulers)
