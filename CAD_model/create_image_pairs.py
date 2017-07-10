#!/usr/bin/env python
"""This script shows an example of using the PyWavefront module."""
import sys
sys.path.append('..')
import ctypes

import pyglet
from pyglet.gl import *
import numpy as np
import pywavefront
import trimesh
import os


def _gl_vector(array, *args):
    '''
    Convert an array and an optional set of args into a flat vector of GLfloat
    '''
    array = np.array(array)
    if len(args) > 0:
        array = np.append(array, args)
    vector = (gl.GLfloat * len(array))(*array)
    return vector

# @window.event
# def on_key_press(symbol, modifiers):
#     if symbol == pyglet.window.key.W:
#         print("here")
#         meshes.draw_texture = not meshes.draw_texture
#
# def update(dt):
#      global rotation
#      rotation += 45*dt
#      if rotation > 720: rotation = 0

def create_image_pairs(filename_list, save_file_list):
    nRotation = 4
    degreePerRotation = 360 // nRotation

    for filename, save_file in zip(filename_list, save_file_list):
        for i in range(nRotation):
            try:
                a = Test(filename, False, save_file, degreePerRotation * i)
                a.save_image()
                b = Test(filename, True, save_file, degreePerRotation * i)
                b.save_image()
            except:
                continue
    
    pyglet.app.run()
    pyglet.app.stop()


lightfv = ctypes.c_float * 4

class Test(object):
    def __init__(self, filename, draw_texture, save_file, degree):
        self.rotation = degree
        self.meshes = pywavefront.Wavefront(filename, draw_texture)
        self.trimesh_object = trimesh.load_mesh(filename)
        self.current_window = pyglet.window.Window(500, 500, resizable=True)
        self.mesh_centroid = self.trimesh_object.centroid
        self.radius = np.max(self.trimesh_object.extents)
        self.draw_texture = draw_texture
        self.save_file = save_file
    def save_image(self):

        #glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60., 1.0, 0.01, 500000.)
        glMatrixMode(GL_MODELVIEW)

        print(self.draw_texture)
        shininess = 128.0
        gl.glClearColor(.93, .93, 1, 1)
        glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        glEnable(GL_LIGHTING)
        glEnable(gl.GL_LIGHT0)
        glEnable(gl.GL_LIGHT1)


        glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, _gl_vector(.5, .5, 1, 0))
        glLightfv(gl.GL_LIGHT0, gl.GL_SPECULAR, _gl_vector(1, 1, 1, 1))
        glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, _gl_vector(1, 1, 1, 1))
        glLightfv(gl.GL_LIGHT1, gl.GL_POSITION, _gl_vector(0, 0, .5, 0))
        glLightfv(gl.GL_LIGHT1, gl.GL_DIFFUSE, _gl_vector(.5, .5, .5, 1))
        glLightfv(gl.GL_LIGHT1, gl.GL_SPECULAR, _gl_vector(1, 1, 1, 1))
        glEnable(GL_LIGHTING)
        glDisable(GL_COLOR_MATERIAL)
        glEnable(GL_DEPTH_TEST)
        glShadeModel(GL_SMOOTH)
        glMatrixMode(GL_MODELVIEW)
        gluLookAt(self.mesh_centroid[0] - 1.5 * self.radius *
                                     np.sin(self.rotation/360.0 * 2 * np.pi),
                  0.5 * self.radius + self.mesh_centroid[1],
                  self.mesh_centroid[2] + 1.5 * self.radius *
                                     np.cos(self.rotation/360.0 * 2 * np.pi),
                  self.mesh_centroid[0],
                  self.mesh_centroid[1],
                  self.mesh_centroid[2],
                  0,
                  1,
                  0)
        self.meshes.draw()
        colorbuffer = pyglet.image.get_buffer_manager().get_color_buffer()
        if self.draw_texture:
            colorbuffer.save("/hdd/Documents/Data/ShapeNetCoreV2/texture_image_4" +
                            self.save_file + "_%d" %self.rotation + ".png")
            #colorbuffer.save("/hdd/Documents/Data/IKEA_PAIR/CAD_Texture_rotation/" +
            #                self.save_file + "_%d" %self.rotation + ".png")
            # colorbuffer.save("/hdd/Documents/Data/IKEA_PAIR/CAD_Texture/" +
            #                  self.save_file + ".png")
            #colorbuffer.save("/home/jiajun/Desktop/1.png")
            self.current_window.close()
            del self.current_window
        else:
            colorbuffer.save("/hdd/Documents/Data/ShapeNetCoreV2/plain_image_4" +
                            self.save_file + "_%d" %self.rotation + ".png")
            #colorbuffer.save("/hdd/Documents/Data/IKEA_PAIR/CAD_Plain_rotation/" +
            #                self.save_file + "_%d" %self.rotation + ".png")
            # colorbuffer.save("/hdd/Documents/Data/IKEA_PAIR/CAD_Plain/" +
            #                  self.save_file + ".png")
            #colorbuffer.save("/home/jiajun/Desktop/2.png")
            self.current_window.close()
            del self.current_window
