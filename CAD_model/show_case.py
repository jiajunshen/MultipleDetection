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

rotation = 0
save_image = False
#meshes = pywavefront.Wavefront('uv_sphere.obj')
IKEA_directory = os.environ['IKEA']
file_name = IKEA_directory + '/IKEA_chair_POANG/nojitter_png_24abfbc0942cbf8fc8b7874340ccdda3_obj0/24abfbc0942cbf8fc8b7874340ccdda3_obj0_object.obj'
#file_name = '/hdd/Documents/Research/IKEA/IKEA_bed_BRIMNES/nojitter_png_4b534bead1e06a7f9ef2df9927efa75_obj0/4b534bead1e06a7f9ef2df9927efa75_obj0_object.obj'
#file_name = 'earth.obj'
#file_name = '/home/jiajun/Dropbox/us_police_car.obj'
#file_name = '/home/jiajun/Dropbox/IRON+MAN+Mark+III.obj'
file_name = '/hdd/Documents/Data/IKEA/IKEA_bed_BRIMNES/nojitter_png_4b534bead1e06a7f9ef2df9927efa75_obj0/4b534bead1e06a7f9ef2df9927efa75_obj0_object.obj'

file_name = '/hdd/Downloads/ShapeNetCore.v2/02691156/5294c39d2a57bd7e5cad6226edb8e82/models/model_normalized.obj'
file_name = '/hdd/Downloads/ShapeNetCore.v2/02691156/73945c1b294716e1d041917556492646/models/model_normalized.obj'
file_name = '/home/jiajun/Desktop/untitled.obj'



meshes = pywavefront.Wavefront(file_name, False)
trimesh_object = trimesh.load_mesh(file_name)
window = pyglet.window.Window(500, 500, resizable=True)
mesh_centroid = trimesh_object.centroid
radius = np.max(trimesh_object.extents)
lightfv = ctypes.c_float * 4

def _gl_vector(array, *args):
    '''
    Convert an array and an optional set of args into a flat vector of GLfloat
    '''
    array = np.array(array)
    if len(args) > 0:
        array = np.append(array, args)
    vector = (gl.GLfloat * len(array))(*array)
    return vector

@window.event
def on_resize(width, height):
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60., float(width)/height, 0.01, 500000.)
    glMatrixMode(GL_MODELVIEW)
    return True


@window.event
def on_draw():
    shininess = 128.0
    gl.glClearColor(.93, .93, 1, 1)
    glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    glEnable(GL_LIGHTING)
    glEnable(gl.GL_LIGHT0)
    glEnable(gl.GL_LIGHT1)


    glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, _gl_vector(.5, .5, 1, 0))
    #glLightfv(gl.GL_LIGHT0, gl.GL_AMBIENT, _gl_vector(.5, .5, .5, 0))
    glLightfv(gl.GL_LIGHT0, gl.GL_SPECULAR, _gl_vector(1, 1, 1, 1))
    glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, _gl_vector(1, 1, 1, 1))
    glLightfv(gl.GL_LIGHT1, gl.GL_POSITION, _gl_vector(0, 0, .5, 0))
    glLightfv(gl.GL_LIGHT1, gl.GL_DIFFUSE, _gl_vector(.5, .5, .5, 1))
    glLightfv(gl.GL_LIGHT1, gl.GL_SPECULAR, _gl_vector(1, 1, 1, 1))

    # #glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, _gl_vector(.5, .5, 1, 0))
    # glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, _gl_vector(1, 1, -.5, 0))
    # #glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, _gl_vector(1, 1, -.5, 0))
    # glLightfv(gl.GL_LIGHT0, gl.GL_SPECULAR, _gl_vector(1, 1, 1, 1))
    # glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, _gl_vector(1, 1, 1, 1))
    # glLightfv(gl.GL_LIGHT1, gl.GL_POSITION, _gl_vector(1, 0, .5, 0))
    # glLightfv(gl.GL_LIGHT1, gl.GL_DIFFUSE, _gl_vector(.5, .5, .5, 1))
    # glLightfv(gl.GL_LIGHT1, gl.GL_SPECULAR, _gl_vector(1, 1, 1, 1))

    # gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, _gl_vector(.5, .5, 1, 0))
    # gl.glLightfv(gl.GL_LIGHT0, gl.GL_SPECULAR, _gl_vector(1, 1, 1, 1))
    # gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, _gl_vector(1, 1, 1, 1))
    # gl.glLightfv(gl.GL_LIGHT1, gl.GL_POSITION, _gl_vector(1, 0, .5, 0))
    # gl.glLightfv(gl.GL_LIGHT1, gl.GL_DIFFUSE, _gl_vector(.5, .5, .5, 1))
    # gl.glLightfv(gl.GL_LIGHT1, gl.GL_SPECULAR, _gl_vector(1, 1, 1, 1))

    #
    # glLightfv(GL_LIGHT0, GL_POSITION, lightfv(-40, 200, 100, 0.0))
    # glLightfv(GL_LIGHT0, GL_AMBIENT, lightfv(0.2, 0.2, 0.2, 1.0))
    # glLightfv(GL_LIGHT0, GL_DIFFUSE, lightfv(0.5, 0.5, 0.5, 1.0))
    #glEnable(GL_LIGHT0)
    glEnable(GL_LIGHTING)

    glDisable(GL_COLOR_MATERIAL)
    glEnable(GL_DEPTH_TEST)
    glDisable(GL_CULL_FACE)
    glShadeModel(GL_SMOOTH)

    glMatrixMode(GL_MODELVIEW)
    #glTranslated(-689,  -295, 0)
    #glRotatef(rotation, 0, 1, 0)
    #gluLookAt(-5255, 0, -832, 0, 0, -832, 0, 1, 0)
    #gluLookAt(-np.sin(rotation/360.0 * 2 * np.pi) * 2450,
    #           0,
    #           np.cos(rotation/360.0 * 2 * np.pi) * 1255 * 2 -1255,
    #           0, 0, -1255, 0, 1, 0)


    gluLookAt(mesh_centroid[0] - 1.5 * radius *
                                 np.sin(rotation/360.0 * 2 * np.pi),
              0.5 * radius + mesh_centroid[1],
              mesh_centroid[2] + 1.5 * radius *
                                 np.cos(rotation/360.0 * 2 * np.pi),
              mesh_centroid[0],
              mesh_centroid[1],
              mesh_centroid[2],
              0,
              1,
              0)
    #glTranslated(0,  0, -6000)
    #glTranslated(-689,  -295, -6000)
    #glRotatef(rotation, 0, 1, 0)
    #glRotatef(0, 1, 0, 0)
    #glRotatef(0, 0, 0, 1)
    #glTranslated(0,  0, -6255.9)
    meshes.draw()
    #glMatrixMode(GL_MODELVIEW)
    if save_image:
        colorbuffer = pyglet.image.get_buffer_manager().get_color_buffer()
        colorbuffer.save("/home/jiajun/Desktop/test_img.png")
        window.close()
@window.event
def on_key_press(symbol, modifiers):
    if symbol == pyglet.window.key.W:
        print("here")
        meshes.draw_texture = not meshes.draw_texture

def update(dt):
     global rotation
     rotation += 45*dt
     if rotation > 720: rotation = 0

pyglet.clock.schedule(update)

pyglet.app.run()
