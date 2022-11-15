import pygame as pg
from OpenGL.GL import *
import numpy as np
import ctypes
from OpenGL.GL.shaders import compileProgram, compileShader


class Renderer:

    def __init__(self, width, height):

        # initialize
        pg.init
        pg.display.set_mode((width, height), pg.OPENGL | pg.DOUBLEBUF)
        self.clock = pg.time.Clock()
        # initialize OpenGL
        glClearColor(0.1, 0.2, 0.2, 1)
        self.shader = self.createShader(
            "shaders/vertex.txt", "shaders/fragment.txt"
        )
        glUseProgram(self.shader)
        self.triangle = Triangle()
        self.mainLoop()

    def mainLoop(self):

        running = True
        while(running):
            # check events
            for event in pg.event.get():
                if (event.type == pg.QUIT):
                    running = False
                elif (event.type == pg.KEYDOWN):
                    if event.key == pg.K_ESCAPE:
                        running = False

            # refresh screen
            glClear(GL_COLOR_BUFFER_BIT)
            pg.display.flip()
            glUseProgram(self.shader)
            glBindVertexArray(self.triangle.vao)
            glDrawArrays(GL_TRIANGLES, 0, self.triangle.vertex_count)

            pg.display.flip()
            # timing
            self.clock.tick(60)

        self.quit()

    def quit(self):
        self.triangle.destroy()
        glDeleteProgram(self.shader)
        pg.quit()

    def createShader(self, vertexFilePath, fragmentFilePath):

        with open(vertexFilePath, 'r') as f:
            vertex_src = f.readlines()
        with open(fragmentFilePath, 'r') as f:
            fragment_src = f.readlines()
        shader = compileProgram(
            compileShader(vertex_src, GL_VERTEX_SHADER),
            compileShader(fragment_src, GL_FRAGMENT_SHADER)
        )
        return shader


class Triangle:
    def __init__(self):
        # Positions
        # x , y , z , s , t , nx ,ny ,nz
        self.vertices = (
            -0.5, -0.5, 0.0, 1.0, 0.0, 0.0,
            0.5, -0.5, 0.0, 0.0, 1.0, 0.0,
            0.0, 0.5, 0.0, 0.0, 0.0, 1.0,
        )
        self.vertices = np.array(self.vertices, dtype=np.float32)
        self.vertex_count = 3
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes,
                     self.vertices, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE,
                              24, ctypes.c_void_p(12))

    def destroy(self):
        glDeleteVertexArrays(1, (self.vao,))
        glDeleteVertexArrays(1, (self.vbo,))


if __name__ == "__main__":
    myRenderer = Renderer(640, 480)
