"""
Welcome to CARLA manual control.

Use ARROWS or WASD keys for control.

    W            : throttle
    S            : brake
    A/D          : steer left/right
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot
    M            : toggle map recording on/off (only works when map is chosen - key 2)
    
    CTRL + W     : toggle constant velocity mode at 60 km/h

    L            : toggle next light type
    SHIFT + L    : toggle high beam
    Z/X          : toggle right/left blinker
    I            : toggle interior light

    TAB          : change sensor position
    ` or N       : next sensor
    [1-9]        : change to sensor [1-9]
    G            : toggle radar visualization
    C            : change weather (Shift+C reverse)
    Backspace    : change vehicle

    O            : open/close all doors of vehicle
    T            : toggle vehicle's telemetry

    V            : Select next map layer (Shift+V reverse)
    B            : Load current selected map layer (Shift+B to unload)

    R            : toggle recording images to disk

    CTRL + R     : toggle recording of simulation (replacing any previous)
    CTRL + P     : start replaying last recorded simulation
    CTRL + +     : increments the start time of the replay by 1 second (+SHIFT = 10 seconds)
    CTRL + -     : decrements the start time of the replay by 1 second (+SHIFT = 10 seconds)

    F1           : toggle HUD
    H/?          : toggle help
    ESC          : quit
"""

from __future__ import print_function


# ==============================================================================
# -- Найдем модуль Carla ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================
sys.path.append('C:/CARLA_0.9.13/PythonAPI/carla')
from agents.navigation.global_route_planner import GlobalRoutePlanner

import carla

from carla import ColorConverter as cc
from carla import TrafficLightState as tls

import argparse
import collections
import datetime
import time
import logging
import math
import random
import re
import weakref
import hashlib

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_b
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_g
    from pygame.locals import K_h
    from pygame.locals import K_i
    from pygame.locals import K_l
    from pygame.locals import K_m
    from pygame.locals import K_n
    from pygame.locals import K_o
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_t
    from pygame.locals import K_v
    from pygame.locals import K_w
    from pygame.locals import K_x
    from pygame.locals import K_z
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

# ==============================================================================
# -- Константы  ----------------------------------------------------------------
# ==============================================================================

# Точка крепления камеры на Tesla Model 3 для получения изображения, близкого к реальному автомобилю
CAMERA_POS_Z = 1.3 
CAMERA_POS_X = 1.4 

CAM_HEIGHT = 480
CAM_WIDTH = 640

# Захват изображения
MAP_FOLDER = 'Map Projection/map_img/'
IMG_FOLDER = 'Map Projection/img/'

MAP_CAPTURE_CROP_FACTOR = 1.5 # на какой коэффициент мы обрезаем высоту и ширину карты, чтобы сохранить ее изображение

# Цвета

# Мы будем использовать цветовую палитру, применяемую в проекте Tango Desktop (каждый цвет индексируется в зависимости от уровня яркости).
# См.: https://en.wikipedia.org/wiki/Tango_Desktop_Project

COLOR_BUTTER_0 = pygame.Color(252, 233, 79)
COLOR_BUTTER_1 = pygame.Color(237, 212, 0)
COLOR_BUTTER_2 = pygame.Color(196, 160, 0)

COLOR_ORANGE_0 = pygame.Color(252, 175, 62)
COLOR_ORANGE_1 = pygame.Color(245, 121, 0)
COLOR_ORANGE_2 = pygame.Color(209, 92, 0)

COLOR_CHOCOLATE_0 = pygame.Color(233, 185, 110)
COLOR_CHOCOLATE_1 = pygame.Color(193, 125, 17)
COLOR_CHOCOLATE_2 = pygame.Color(143, 89, 2)

COLOR_CHAMELEON_0 = pygame.Color(138, 226, 52)
COLOR_CHAMELEON_1 = pygame.Color(115, 210, 22)
COLOR_CHAMELEON_2 = pygame.Color(78, 154, 6)

COLOR_SKY_BLUE_0 = pygame.Color(114, 159, 207)
COLOR_SKY_BLUE_1 = pygame.Color(52, 101, 164)
COLOR_SKY_BLUE_2 = pygame.Color(32, 74, 135)

COLOR_PLUM_0 = pygame.Color(173, 127, 168)
COLOR_PLUM_1 = pygame.Color(117, 80, 123)
COLOR_PLUM_2 = pygame.Color(92, 53, 102)

COLOR_SCARLET_RED_0 = pygame.Color(239, 41, 41)
COLOR_SCARLET_RED_1 = pygame.Color(204, 0, 0)
COLOR_SCARLET_RED_2 = pygame.Color(164, 0, 0)

COLOR_ALUMINIUM_0 = pygame.Color(238, 238, 236)
COLOR_ALUMINIUM_1 = pygame.Color(211, 215, 207)
COLOR_ALUMINIUM_2 = pygame.Color(186, 189, 182)
COLOR_ALUMINIUM_3 = pygame.Color(136, 138, 133)
COLOR_ALUMINIUM_4 = pygame.Color(85, 87, 83)
COLOR_ALUMINIUM_4_5 = pygame.Color(66, 62, 64)
COLOR_ALUMINIUM_5 = pygame.Color(46, 52, 54)


COLOR_WHITE = pygame.Color(255, 255, 255)
COLOR_BLACK = pygame.Color(0, 0, 0)

# Module Defines
TITLE_WORLD = 'WORLD'
TITLE_HUD = 'HUD'
TITLE_INPUT = 'INPUT'

PIXELS_PER_METER = 12

MAP_DEFAULT_SCALE = 0.1
HERO_DEFAULT_SCALE = 1.0

PIXELS_AHEAD_VEHICLE = 150

# ==============================================================================
# -- Глобальные функции ----------------------------------------------------------
# ==============================================================================


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        if int_generation in [1, 2]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []

# ==============================================================================
# -- Утилита скопирована из примера "no_rendereing".---------------------------------
# ==============================================================================


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


class Util(object):

    @staticmethod
    def blits(destination_surface, source_surfaces, rect=None, blend_mode=0):
        """Функция, которая рендерит все исходные поверхности в целевом источнике"""
        for surface in source_surfaces:
            destination_surface.blit(surface[0], surface[1], rect, blend_mode)

    @staticmethod
    def length(v):
        """Возвращает длину вектора"""
        return math.sqrt(v.x**2 + v.y**2 + v.z**2)

    @staticmethod
    def get_bounding_box(actor):
        """Получает углы ограничительной рамки агента в глобальном пространстве"""
        bb = actor.trigger_volume.extent
        corners = [carla.Location(x=-bb.x, y=-bb.y),
                   carla.Location(x=bb.x, y=-bb.y),
                   carla.Location(x=bb.x, y=bb.y),
                   carla.Location(x=-bb.x, y=bb.y),
                   carla.Location(x=-bb.x, y=-bb.y)]
        corners = [x + actor.trigger_volume.location for x in corners]
        t = actor.get_transform()
        t.transform(corners)
        return corners

# ==============================================================================
# -- Светофоры ------------------------------------------------------
# ==============================================================================


class TrafficLightSurfaces(object):
    """Содержит поверхности (масштабированные и развернутые) для отображения светофоров"""

    def __init__(self):
        def make_surface(tl):
            """Изображает светофор, состоящий из темной фоновой поверхности с 3 кругами, которые указывают на его цвет в зависимости от состояния"""
            w = 40
            surface = pygame.Surface((w, 3 * w), pygame.SRCALPHA)
            surface.fill(COLOR_ALUMINIUM_5 if tl != 'h' else COLOR_ORANGE_2)
            if tl != 'h':
                hw = int(w / 2)
                off = COLOR_ALUMINIUM_4
                red = COLOR_SCARLET_RED_0
                yellow = COLOR_BUTTER_0
                green = COLOR_CHAMELEON_0

                # Окрашивает в соответствующий цвет, если он включен, в противном случае он будет серым, если выключен
                pygame.draw.circle(surface, red if tl == tls.Red else off, (hw, hw), int(0.4 * w))
                pygame.draw.circle(surface, yellow if tl == tls.Yellow else off, (hw, w + hw), int(0.4 * w))
                pygame.draw.circle(surface, green if tl == tls.Green else off, (hw, 2 * w + hw), int(0.4 * w))

            return pygame.transform.smoothscale(surface, (15, 45) if tl != 'h' else (19, 49))

        self._original_surfaces = {
            'h': make_surface('h'),
            tls.Red: make_surface(tls.Red),
            tls.Yellow: make_surface(tls.Yellow),
            tls.Green: make_surface(tls.Green),
            tls.Off: make_surface(tls.Off),
            tls.Unknown: make_surface(tls.Unknown)
        }
        self.surfaces = dict(self._original_surfaces)

    def rotozoom(self, angle, scale):
        for key, surface in self._original_surfaces.items():
            self.surfaces[key] = pygame.transform.rotozoom(surface, angle, scale)


class MapImage(object):
    """Класс предназначен для рендеринга 2D-изображения с видом сверху на мир Карла.
    Обратите внимание, что используется система кэширования, поэтому если содержимое OpenDrive
    города Carla не изменилось, то будет прочитано и использовано сохраненное изображение, если оно было отрисовано в предыдущем выполнении."""

    def __init__(self, carla_world, carla_map, pixels_per_meter, show_triggers, show_connections, show_spawn_points):
        """ Рендеринг изображения карты, созданного на основе мира, его карты и дополнительных флагов,
        которые предоставляют дополнительную информацию о дорожной сети"""
        self._pixels_per_meter = pixels_per_meter
        self.scale = 1.0
        self.show_triggers = show_triggers
        self.show_connections = show_connections
        self.show_spawn_points = show_spawn_points

        waypoints = carla_map.generate_waypoints(2)
        margin = 50
        max_x = max(waypoints, key=lambda x: x.transform.location.x).transform.location.x + margin
        max_y = max(waypoints, key=lambda x: x.transform.location.y).transform.location.y + margin
        min_x = min(waypoints, key=lambda x: x.transform.location.x).transform.location.x - margin
        min_y = min(waypoints, key=lambda x: x.transform.location.y).transform.location.y - margin

        self.width = max(max_x - min_x, max_y - min_y)
        self._world_offset = (min_x, min_y)

        # Максимальный размер изображения Pygame
        width_in_pixels = (1 << 14) - 1

        # Адаптируем пиксели на метр, чтобы мир поместился на изображении
        surface_pixel_per_meter = int(width_in_pixels / self.width)
        if surface_pixel_per_meter > PIXELS_PER_METER:
            surface_pixel_per_meter = PIXELS_PER_METER

        self._pixels_per_meter = surface_pixel_per_meter
        width_in_pixels = int(self._pixels_per_meter * self.width)

        self.big_map_surface = pygame.Surface((width_in_pixels, width_in_pixels)).convert()

        # Загрузка содержимого OpenDrive
        opendrive_content = carla_map.to_opendrive()

        # Получение хэша на основе содержимого
        hash_func = hashlib.sha1()
        hash_func.update(opendrive_content.encode("UTF-8"))
        opendrive_hash = str(hash_func.hexdigest())

        # Путь для сохранения или загрузки кэшированной карты
        filename = carla_map.name.split('/')[-1] + "_" + opendrive_hash + ".tga"
        dirname = os.path.join("cache", "no_rendering_mode")
        full_path = str(os.path.join(dirname, filename))

        if os.path.isfile(full_path):
            # Загрузка изображения
            self.big_map_surface = pygame.image.load(full_path)
        else:
            self.draw_road_map(
                self.big_map_surface,
                carla_world,
                carla_map,
                self.world_to_pixel,
                self.world_to_pixel_width)

            # Если путь к папкам не существует, создаем его
            if not os.path.exists(dirname):
                os.makedirs(dirname)

            # Удаление файлов, если в выбранном городе была сохранена предыдущая версия
            list_filenames = glob.glob(os.path.join(dirname, carla_map.name) + "*")
            for town_filename in list_filenames:
                os.remove(town_filename)

            # Сохранение отрисованной карты для последующих исполнений той же карты
            pygame.image.save(self.big_map_surface, full_path)

        self.surface = self.big_map_surface

    def draw_road_map(self, map_surface, carla_world, carla_map, world_to_pixel, world_to_pixel_width):
        """Отрисовывает все дороги, включая разметку полос движения, указатели и дорожные знаки"""
        map_surface.fill(COLOR_ALUMINIUM_4)
        precision = 0.05

        def lane_marking_color_to_tango(lane_marking_color):
            """Сопоставляет перечисление цветов разметки полосы движения, указанное в PythonAPI, с цветом Tango Color"""
            tango_color = COLOR_BLACK

            if lane_marking_color == carla.LaneMarkingColor.White:
                tango_color = COLOR_ALUMINIUM_2

            elif lane_marking_color == carla.LaneMarkingColor.Blue:
                tango_color = COLOR_SKY_BLUE_0

            elif lane_marking_color == carla.LaneMarkingColor.Green:
                tango_color = COLOR_CHAMELEON_0

            elif lane_marking_color == carla.LaneMarkingColor.Red:
                tango_color = COLOR_SCARLET_RED_0

            elif lane_marking_color == carla.LaneMarkingColor.Yellow:
                tango_color = COLOR_ORANGE_0

            return tango_color

        def draw_solid_line(surface, color, closed, points, width):
            """Проводит сплошные линии на поверхности, задавая набор точек, ширину и цвет"""
            if len(points) >= 2:
                pygame.draw.lines(surface, color, closed, points, width)

        def draw_broken_line(surface, color, closed, points, width):
            """Чертит ломаные линии на поверхности с заданным набором точек, шириной и цветом."""
            # Выбираем, какие линии будут отрисованы из набора линий
            broken_lines = [x for n, x in enumerate(zip(*(iter(points),) * 20)) if n % 3 == 0]

            # Чертит выбранные линии
            for line in broken_lines:
                pygame.draw.lines(surface, color, closed, line, width)

        def get_lane_markings(lane_marking_type, lane_marking_color, waypoints, sign):
            """Для нескольких типов разметки полосы движения (SolidSolid, BrokenSolid, SolidBroken и BrokenBroken)
            преобразует их как комбинацию прерывистых и сплошных линий"""
            margin = 0.25
            marking_1 = [world_to_pixel(lateral_shift(w.transform, sign * w.lane_width * 0.5)) for w in waypoints]
            if lane_marking_type == carla.LaneMarkingType.Broken or (lane_marking_type == carla.LaneMarkingType.Solid):
                return [(lane_marking_type, lane_marking_color, marking_1)]
            else:
                marking_2 = [world_to_pixel(lateral_shift(w.transform,
                                                          sign * (w.lane_width * 0.5 + margin * 2))) for w in waypoints]
                if lane_marking_type == carla.LaneMarkingType.SolidBroken:
                    return [(carla.LaneMarkingType.Broken, lane_marking_color, marking_1),
                            (carla.LaneMarkingType.Solid, lane_marking_color, marking_2)]
                elif lane_marking_type == carla.LaneMarkingType.BrokenSolid:
                    return [(carla.LaneMarkingType.Solid, lane_marking_color, marking_1),
                            (carla.LaneMarkingType.Broken, lane_marking_color, marking_2)]
                elif lane_marking_type == carla.LaneMarkingType.BrokenBroken:
                    return [(carla.LaneMarkingType.Broken, lane_marking_color, marking_1),
                            (carla.LaneMarkingType.Broken, lane_marking_color, marking_2)]
                elif lane_marking_type == carla.LaneMarkingType.SolidSolid:
                    return [(carla.LaneMarkingType.Solid, lane_marking_color, marking_1),
                            (carla.LaneMarkingType.Solid, lane_marking_color, marking_2)]

            return [(carla.LaneMarkingType.NONE, carla.LaneMarkingColor.Other, [])]

        def draw_lane(surface, lane, color):
            """Отображает одну полосу на поверхности с заданным цветом"""
            for side in lane:
                lane_left_side = [lateral_shift(w.transform, -w.lane_width * 0.5) for w in side]
                lane_right_side = [lateral_shift(w.transform, w.lane_width * 0.5) for w in side]

                polygon = lane_left_side + [x for x in reversed(lane_right_side)]
                polygon = [world_to_pixel(x) for x in polygon]

                if len(polygon) > 2:
                    pygame.draw.polygon(surface, color, polygon, 5)
                    pygame.draw.polygon(surface, color, polygon)

        def draw_lane_marking(surface, waypoints):
            """Отрисовывает левую и правую стороны разметки полосы движения"""
            # Левая сторона
            draw_lane_marking_single_side(surface, waypoints[0], -1)

            # Правая сторона
            draw_lane_marking_single_side(surface, waypoints[1], 1)

        def draw_lane_marking_single_side(surface, waypoints, sign):
            """Рисует дорожную разметку, заданную набором путевых точек,
            и решает, рисовать ли правую или левую сторону от путевой точки"""
            lane_marking = None

            marking_type = carla.LaneMarkingType.NONE
            previous_marking_type = carla.LaneMarkingType.NONE

            marking_color = carla.LaneMarkingColor.Other
            previous_marking_color = carla.LaneMarkingColor.Other

            markings_list = []
            temp_waypoints = []
            current_lane_marking = carla.LaneMarkingType.NONE
            for sample in waypoints:
                lane_marking = sample.left_lane_marking if sign < 0 else sample.right_lane_marking

                if lane_marking is None:
                    continue

                marking_type = lane_marking.type
                marking_color = lane_marking.color

                if current_lane_marking != marking_type:
                    # Получение списка разметки для отрисовки полос движения
                    markings = get_lane_markings(
                        previous_marking_type,
                        lane_marking_color_to_tango(previous_marking_color),
                        temp_waypoints,
                        sign)
                    current_lane_marking = marking_type

                    # Добавляем каждую разметку полосы движения в список
                    for marking in markings:
                        markings_list.append(marking)

                    temp_waypoints = temp_waypoints[-1:]

                else:
                    temp_waypoints.append((sample))
                    previous_marking_type = marking_type
                    previous_marking_color = marking_color

            # Добавление последней маркировки
            last_markings = get_lane_markings(
                previous_marking_type,
                lane_marking_color_to_tango(previous_marking_color),
                temp_waypoints,
                sign)
            for marking in last_markings:
                markings_list.append(marking)

            # После того как разметка полос движения была упрощена до сплошных или прерывистых линий, мы отрисовываем их
            for markings in markings_list:
                if markings[0] == carla.LaneMarkingType.Solid:
                    draw_solid_line(surface, markings[1], False, markings[2], 2)
                elif markings[0] == carla.LaneMarkingType.Broken:
                    draw_broken_line(surface, markings[1], False, markings[2], 2)

        def draw_arrow(surface, transform, color=COLOR_ALUMINIUM_2):
            """ Строит стрелку заданного цвета с учетом трансформации"""
            transform.rotation.yaw += 180
            forward = transform.get_forward_vector()
            transform.rotation.yaw += 90
            right_dir = transform.get_forward_vector()
            end = transform.location
            start = end - 2.0 * forward
            right = start + 0.8 * forward + 0.4 * right_dir
            left = start + 0.8 * forward - 0.4 * right_dir

            # Рисуем линии
            pygame.draw.lines(surface, color, False, [world_to_pixel(x) for x in [start, end]], 4)
            pygame.draw.lines(surface, color, False, [world_to_pixel(x) for x in [left, start, right]], 4)

        def draw_traffic_signs(surface, font_surface, actor, color=COLOR_ALUMINIUM_2, trigger_color=COLOR_PLUM_0):
            """Отображает знаки остановки транспорта"""
            transform = actor.get_transform()
            waypoint = carla_map.get_waypoint(transform.location)

            angle = -waypoint.transform.rotation.yaw - 90.0
            font_surface = pygame.transform.rotate(font_surface, angle)
            pixel_pos = world_to_pixel(waypoint.transform.location)
            offset = font_surface.get_rect(center=(pixel_pos[0], pixel_pos[1]))
            surface.blit(font_surface, offset)

            forward_vector = carla.Location(waypoint.transform.get_forward_vector())
            left_vector = carla.Location(-forward_vector.y, forward_vector.x,
                                         forward_vector.z) * waypoint.lane_width / 2 * 0.7

            line = [(waypoint.transform.location + (forward_vector * 1.5) + (left_vector)),
                    (waypoint.transform.location + (forward_vector * 1.5) - (left_vector))]

            line_pixel = [world_to_pixel(p) for p in line]
            pygame.draw.lines(surface, color, True, line_pixel, 2)

            # Построение ограничительной рамки срабатывания остановки
            if self.show_triggers:
                corners = Util.get_bounding_box(actor)
                corners = [world_to_pixel(p) for p in corners]
                pygame.draw.lines(surface, trigger_color, True, corners, 2)


        def lateral_shift(transform, shift):
            """Выполняет боковой сдвиг прямого вектора преобразования"""
            transform.rotation.yaw += 90
            return transform.location + shift * transform.get_forward_vector()

        def draw_topology(carla_topology, index):
            """Рисует дорожные знаки и сеть дорог с тротуарами, парковками и обочинами, генерируя путевые точки"""
            topology = [x[index] for x in carla_topology]
            topology = sorted(topology, key=lambda w: w.transform.location.z)
            set_waypoints = []
            for waypoint in topology:
                waypoints = [waypoint]

                # Генерирование путевых точек по идентификатору дороги. Остановка при отличии идентификатора дороги
                nxt = waypoint.next(precision)
                if len(nxt) > 0:
                    nxt = nxt[0]
                    while nxt.road_id == waypoint.road_id:
                        waypoints.append(nxt)
                        nxt = nxt.next(precision)
                        if len(nxt) > 0:
                            nxt = nxt[0]
                        else:
                            break
                set_waypoints.append(waypoints)

                # Отображение обочин, парковок и тротуаров
                PARKING_COLOR = COLOR_ALUMINIUM_4_5
                SHOULDER_COLOR = COLOR_ALUMINIUM_5
                SIDEWALK_COLOR = COLOR_ALUMINIUM_3

                shoulder = [[], []]
                parking = [[], []]
                sidewalk = [[], []]

                for w in waypoints:
                    # Классифицировать типы полос движения, пока не останется ни одной путевой точки, перемещаясь влево
                    l = w.get_left_lane()
                    while l and l.lane_type != carla.LaneType.Driving:

                        if l.lane_type == carla.LaneType.Shoulder:
                            shoulder[0].append(l)

                        if l.lane_type == carla.LaneType.Parking:
                            parking[0].append(l)

                        if l.lane_type == carla.LaneType.Sidewalk:
                            sidewalk[0].append(l)

                        l = l.get_left_lane()

                    # Классифицировать типы полос движения, пока не останется ни одной путевой точки, перемещаясь вправо
                    r = w.get_right_lane()
                    while r and r.lane_type != carla.LaneType.Driving:

                        if r.lane_type == carla.LaneType.Shoulder:
                            shoulder[1].append(r)

                        if r.lane_type == carla.LaneType.Parking:
                            parking[1].append(r)

                        if r.lane_type == carla.LaneType.Sidewalk:
                            sidewalk[1].append(r)

                        r = r.get_right_lane()

                # Отображает типы полос движения
                draw_lane(map_surface, shoulder, SHOULDER_COLOR)
                draw_lane(map_surface, parking, PARKING_COLOR)
                draw_lane(map_surface, sidewalk, SIDEWALK_COLOR)

            # Изобржаем дороги
            for waypoints in set_waypoints:
                waypoint = waypoints[0]
                road_left_side = [lateral_shift(w.transform, -w.lane_width * 0.5) for w in waypoints]
                road_right_side = [lateral_shift(w.transform, w.lane_width * 0.5) for w in waypoints]

                polygon = road_left_side + [x for x in reversed(road_right_side)]
                polygon = [world_to_pixel(x) for x in polygon]

                if len(polygon) > 2:
                    pygame.draw.polygon(map_surface, COLOR_ALUMINIUM_5, polygon, 5)
                    pygame.draw.polygon(map_surface, COLOR_ALUMINIUM_5, polygon)

                # Изобржаем разметку и стрелки
                if not waypoint.is_junction:
                    draw_lane_marking(map_surface, [waypoints, waypoints])
                    for n, wp in enumerate(waypoints):
                        if ((n + 1) % 400) == 0:
                            draw_arrow(map_surface, wp.transform)

        topology = carla_map.get_topology()
        draw_topology(topology, 0)

        if self.show_spawn_points:
            for sp in carla_map.get_spawn_points():
                draw_arrow(map_surface, sp, color=COLOR_CHOCOLATE_0)

        if self.show_connections:
            dist = 1.5

            def to_pixel(wp): return world_to_pixel(wp.transform.location)
            for wp in carla_map.generate_waypoints(dist):
                col = (0, 255, 255) if wp.is_junction else (0, 255, 0)
                for nxt in wp.next(dist):
                    pygame.draw.line(map_surface, col, to_pixel(wp), to_pixel(nxt), 2)
                if wp.lane_change & carla.LaneChange.Right:
                    r = wp.get_right_lane()
                    if r and r.lane_type == carla.LaneType.Driving:
                        pygame.draw.line(map_surface, col, to_pixel(wp), to_pixel(r), 2)
                if wp.lane_change & carla.LaneChange.Left:
                    l = wp.get_left_lane()
                    if l and l.lane_type == carla.LaneType.Driving:
                        pygame.draw.line(map_surface, col, to_pixel(wp), to_pixel(l), 2)

        actors = carla_world.get_actors()

        # Находим и изображаем дорожные знаки
        font_size = world_to_pixel_width(1)
        font = pygame.font.SysFont('Arial', font_size, True)

        stops = [actor for actor in actors if 'stop' in actor.type_id]
        yields = [actor for actor in actors if 'yield' in actor.type_id]

        stop_font_surface = font.render("STOP", False, COLOR_ALUMINIUM_2)
        stop_font_surface = pygame.transform.scale(
            stop_font_surface, (stop_font_surface.get_width(), stop_font_surface.get_height() * 2))

        yield_font_surface = font.render("YIELD", False, COLOR_ALUMINIUM_2)
        yield_font_surface = pygame.transform.scale(
            yield_font_surface, (yield_font_surface.get_width(), yield_font_surface.get_height() * 2))

        for ts_stop in stops:
            draw_traffic_signs(map_surface, stop_font_surface, ts_stop, trigger_color=COLOR_SCARLET_RED_1)

        for ts_yield in yields:
            draw_traffic_signs(map_surface, yield_font_surface, ts_yield, trigger_color=COLOR_ORANGE_1)

    def world_to_pixel(self, location, offset=(0, 0)):
        """Преобразует глобальные координаты в координаты пикселей"""
        x = self.scale * self._pixels_per_meter * (location.x - self._world_offset[0])
        y = self.scale * self._pixels_per_meter * (location.y - self._world_offset[1])
        return [int(x - offset[0]), int(y - offset[1])]

    def world_to_pixel_width(self, width):
        """Преобразует единицы измерения мира в единицы измерения пикселей"""
        return int(self.scale * self._pixels_per_meter * width)

    def scale_map(self, scale):
        """Масштабирует поверхность карты"""
        if scale != self.scale:
            self.scale = scale
            width = int(self.big_map_surface.get_width() * self.scale)
            self.surface = pygame.transform.smoothscale(self.big_map_surface, (width, width))


# ==============================================================================
# -- Мир ---------------------------------------------------------------------
# ==============================================================================


class World(object):
    def __init__(self, carla_world, hud, args):
        self.world = carla_world
        self.args = args
        self.sync = args.sync
        self.map_capture = args.map_capture
        self.actor_role_name = args.rolename
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.imu_sensor = None
        self.radar_sensor = None
        self.camera_manager = None
        if args.clear:
            for sensor in self.world.get_actors().filter('*sensor*'):
                sensor.destroy()
            for actor in self.world.get_actors().filter('*vehicle*'):
                actor.destroy()
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self._actor_generation = args.generation
        self._gamma = args.gamma

        # Атрибуты отображения из no_rendering
        self.surface_size = [0, 0]
        self.prev_scaled_size = 0
        self.scaled_size = 0

        self.player = None
        self.spawned_hero = None
        self.hero_transform = None

        self.scale_offset = [0, 0]

        self.vehicle_id_surface = None
        self.result_surface = None

        self.traffic_light_surfaces = TrafficLightSurfaces()
        self.affected_traffic_light = None
        self.route = None 
        # Информация о карте
        self.map_image = None
        self.border_round_surface = None
        self.original_surface_size = None
        self.hero_surface = None
        self.actors_surface = None

        self.restart()
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0
        self.constant_velocity_enabled = False
        self.show_vehicle_telemetry = False
        self.doors_are_open = False
        self.current_map_layer = 0
        self.map_layer_names = [
            carla.MapLayer.NONE,
            carla.MapLayer.Buildings,
            carla.MapLayer.Decals,
            carla.MapLayer.Foliage,
            carla.MapLayer.Ground,
            carla.MapLayer.ParkedVehicles,
            carla.MapLayer.Particles,
            carla.MapLayer.Props,
            carla.MapLayer.StreetLights,
            carla.MapLayer.Walls,
            carla.MapLayer.All
        ]

    def restart(self):
        self.player_max_speed = 1.589
        self.player_max_speed_fast = 3.713
        # Сохраняйет конфигурацию камеры, если менеджер камер существует.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        
        blueprint = self.world.get_blueprint_library().filter('*model3*')[0]
        
        blueprint.set_attribute('role_name', self.actor_role_name)
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'true')
        # задаем максимальную скорость
        if blueprint.has_attribute('speed'):
            self.player_max_speed = float(blueprint.get_attribute('speed').recommended_values[1])
            self.player_max_speed_fast = float(blueprint.get_attribute('speed').recommended_values[2])

        
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.show_vehicle_telemetry = False
            self.modify_vehicle_physics(self.player)
        while self.player is None:
            if not self.map.get_spawn_points():
                print('There are no spawn points available in your map/town.')
                print('Please add some Vehicle Spawn Point to your UE4 scene.')
                sys.exit(1)
            spawn_points = self.map.get_spawn_points()
            spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.show_vehicle_telemetry = False
            self.modify_vehicle_physics(self.player)
        
        # Настройка отображения - скопировано из no_rendering
        
        self.map_image = MapImage(
            carla_world=self.world,
            carla_map=self.map,
            pixels_per_meter=PIXELS_PER_METER,
            show_triggers=self.args.show_triggers,
            show_connections=self.args.show_connections,
            show_spawn_points=self.args.show_spawn_points)

        self.original_surface_size = min(self.hud.dim[0], self.hud.dim[1])
        self.surface_size = self.map_image.big_map_surface.get_width()

        self.scaled_size = int(self.surface_size)
        self.prev_scaled_size = int(self.surface_size)

        self.actors_surface = pygame.Surface((self.map_image.surface.get_width(), self.map_image.surface.get_height()))
        self.actors_surface.set_colorkey(COLOR_BLACK)

        self.vehicle_id_surface = pygame.Surface((self.surface_size, self.surface_size)).convert()
        self.vehicle_id_surface.set_colorkey(COLOR_BLACK)

        self.border_round_surface = pygame.Surface(self.hud.dim, pygame.SRCALPHA).convert()
        self.border_round_surface.set_colorkey(COLOR_WHITE)
        self.border_round_surface.fill(COLOR_BLACK)

        # Используется в режиме "героя", рисует карту, заключенную в круг с белой рамкой
        center_offset = (int(self.hud.dim[0] / 2), int(self.hud.dim[1] / 2))
        pygame.draw.circle(self.border_round_surface, COLOR_ALUMINIUM_1, center_offset, int(self.hud.dim[1] / 2))
        pygame.draw.circle(self.border_round_surface, COLOR_WHITE, center_offset, int((self.hud.dim[1] - 8) / 2))

        scaled_original_size = self.original_surface_size * (1.0 / 0.9)
        self.hero_surface = pygame.Surface((scaled_original_size, scaled_original_size)).convert()

        self.result_surface = pygame.Surface((self.surface_size, self.surface_size)).convert()
        self.result_surface.set_colorkey(COLOR_BLACK)

        self.player.set_autopilot(False)

        # Настройка датчиков.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.imu_sensor = IMUSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

        if self.sync:
            self.world.tick()
        else:
            self.world.wait_for_tick()

    def select_random_route(self):
        '''
        прокладывает случайный маршрут для автомобиля от его текущего местоположения
        на возможные места (точки спавна) 
        если расстояние превышает 100 путевых точек
        '''    
        point_a = self.player.get_transform().location # начинаем с того места, где находится автомобиль, или с последней путевой точки
        sampling_resolution = 1
        grp = GlobalRoutePlanner(self.map, sampling_resolution)
        # Теперь давайте выберем самый длинный из возможных маршрутов
        min_distance = 100
        route_list = []
        for loc in self.map.get_spawn_points(): # начинаем пробовать все точки спавна  
                                                # исключаем первый с нулевым индексом
            cur_route = grp.trace_route(point_a, loc.location)
            if len(cur_route) > min_distance:
                route_list.append(cur_route)
        self.route = random.choice(route_list)

    def _show_nearby_vehicles(self, vehicles):
        """Показывает близлежащие автомобили"""
        info_text = []
        if self.player is not None and len(vehicles) > 1:
            location = self.hero_transform.location
            vehicle_list = [x[0] for x in vehicles if x[0].id != self.player.id]

            def distance(v): return location.distance(v.get_location())
            for n, vehicle in enumerate(sorted(vehicle_list, key=distance)):
                if n > 15:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                info_text.append('% 5d %s' % (vehicle.id, vehicle_type))
        

    def _split_actors(self):
        """Разделяет найденные агенты по идентификатору типа"""
        vehicles = []
        traffic_lights = []
        speed_limits = []
        walkers = []

        for actor_with_transform in self.actors_with_transforms:
            actor = actor_with_transform[0]
            if 'vehicle' in actor.type_id:
                vehicles.append(actor_with_transform)
            elif 'traffic_light' in actor.type_id:
                traffic_lights.append(actor_with_transform)
            elif 'speed_limit' in actor.type_id:
                speed_limits.append(actor_with_transform)
            elif 'walker.pedestrian' in actor.type_id:
                walkers.append(actor_with_transform)

        return (vehicles, traffic_lights, speed_limits, walkers)

    def _render_traffic_lights(self, surface, list_tl, world_to_pixel):
        """Отрисовывает светофор и показывает его триггеры и ограничительные рамки, если флаги включены"""
        self.affected_traffic_light = None

        for tl in list_tl:
            world_pos = tl.get_location()
            pos = world_to_pixel(world_pos)

            if self.args.show_triggers:
                corners = Util.get_bounding_box(tl)
                corners = [world_to_pixel(p) for p in corners]
                pygame.draw.lines(surface, COLOR_BUTTER_1, True, corners, 2)

            if self.player is not None:
                corners = Util.get_bounding_box(tl)
                corners = [world_to_pixel(p) for p in corners]
                tl_t = tl.get_transform()

                transformed_tv = tl_t.transform(tl.trigger_volume.location)
                hero_location = self.player.get_location()
                d = hero_location.distance(transformed_tv)
                s = Util.length(tl.trigger_volume.extent) + Util.length(self.player.bounding_box.extent)
                if (d <= s):
                    self.affected_traffic_light = tl
                    srf = self.traffic_light_surfaces.surfaces['h']
                    surface.blit(srf, srf.get_rect(center=pos))

            srf = self.traffic_light_surfaces.surfaces[tl.state]
            surface.blit(srf, srf.get_rect(center=pos))

    def _render_speed_limits(self, surface, list_sl, world_to_pixel, world_to_pixel_width):
        """Отображает ограничения скорости, рисуя два одинаковых круга (внешний - красный, внутренний - белый) и сообщение об ограничении скорости."""

        font_size = world_to_pixel_width(2)
        radius = world_to_pixel_width(2)
        font = pygame.font.SysFont('Arial', font_size)

        for sl in list_sl:

            x, y = world_to_pixel(sl.get_location())

            # Изображение ограничения скорости в виде кругов
            white_circle_radius = int(radius * 0.75)

            pygame.draw.circle(surface, COLOR_SCARLET_RED_1, (x, y), radius)
            pygame.draw.circle(surface, COLOR_ALUMINIUM_0, (x, y), white_circle_radius)

            limit = sl.type_id.split('.')[2]
            font_surface = font.render(limit, True, COLOR_ALUMINIUM_5)

            if self.args.show_triggers:
                corners = Util.get_bounding_box(sl)
                corners = [world_to_pixel(p) for p in corners]
                pygame.draw.lines(surface, COLOR_PLUM_2, True, corners, 2)

            if self.player is not None:
                angle = -self.hero_transform.rotation.yaw - 90.0
                font_surface = pygame.transform.rotate(font_surface, angle)
                offset = font_surface.get_rect(center=(x, y))
                surface.blit(font_surface, offset)

            else:
                # В режиме карты нет необходимости поворачивать текст ограничения скорости
                surface.blit(font_surface, (x - radius / 2, y - radius / 2))

    def _render_walkers(self, surface, list_w, world_to_pixel):
        """Отрисовывает ограничительные рамки пешеходов"""
        for w in list_w:
            color = COLOR_PLUM_0

            # Вычисление точек ограничительных границ
            bb = w[0].bounding_box.extent
            corners = [
                carla.Location(x=-bb.x, y=-bb.y),
                carla.Location(x=bb.x, y=-bb.y),
                carla.Location(x=bb.x, y=bb.y),
                carla.Location(x=-bb.x, y=bb.y)]

            w[1].transform(corners)
            corners = [world_to_pixel(p) for p in corners]
            pygame.draw.polygon(surface, color, corners)

    def _render_vehicles(self, surface, list_v, world_to_pixel):
        """Отрисовывает ограничительные рамки автомобилей"""
        for v in list_v:
            color = COLOR_SKY_BLUE_0
            if int(v[0].attributes['number_of_wheels']) == 2:
                color = COLOR_CHOCOLATE_1
            if v[0].attributes['role_name'] == 'hero':
                color = COLOR_CHAMELEON_0
            # Вычисление точек ограничительной рамки
            bb = v[0].bounding_box.extent
            corners = [carla.Location(x=-bb.x, y=-bb.y),
                       carla.Location(x=bb.x - 0.8, y=-bb.y),
                       carla.Location(x=bb.x, y=0),
                       carla.Location(x=bb.x - 0.8, y=bb.y),
                       carla.Location(x=-bb.x, y=bb.y),
                       carla.Location(x=-bb.x, y=-bb.y)
                       ]
            v[1].transform(corners)
            corners = [world_to_pixel(p) for p in corners]
            pygame.draw.lines(surface, color, False, corners, int(math.ceil(4.0 * self.map_image.scale)))

    def render_actors(self, surface, vehicles, traffic_lights, speed_limits, walkers):
        """Рендеринг всех участников"""
        # Статические участники
        self._render_traffic_lights(surface, [tl[0] for tl in traffic_lights], self.map_image.world_to_pixel)
        self._render_speed_limits(surface, [sl[0] for sl in speed_limits], self.map_image.world_to_pixel,
                                  self.map_image.world_to_pixel_width)

        # Динамические участники
        self._render_vehicles(surface, vehicles, self.map_image.world_to_pixel)
        self._render_walkers(surface, walkers, self.map_image.world_to_pixel)

    def clip_surfaces(self, clipping_rect):
        """Используется для повышения производительности. Обрезает поверхности, чтобы отрисовать только ту часть поверхностей, которая будет видна"""
        self.actors_surface.set_clip(clipping_rect)
        self.vehicle_id_surface.set_clip(clipping_rect)
        self.result_surface.set_clip(clipping_rect)

    def _compute_scale(self, scale_factor):
        """Основываясь на колесике мыши и положении мыши, Функция вычислит масштаб и переместит карту так,
        чтобы она была увеличена или уменьшена в зависимости от положения мыши."""
        m = (0,0)

        # Процент поверхности, на которой фактически находится позиция мыши
        px = (m[0] - self.scale_offset[0]) / float(self.prev_scaled_size)
        py = (m[1] - self.scale_offset[1]) / float(self.prev_scaled_size)

        # Смещение будет ранее накопленным смещением, добавленным с помощью
        # разницы в положении мыши в старой и новой шкалах
        diff_between_scales = ((float(self.prev_scaled_size) * px) - (float(self.scaled_size) * px),
                               (float(self.prev_scaled_size) * py) - (float(self.scaled_size) * py))

        self.scale_offset = (self.scale_offset[0] + diff_between_scales[0],
                             self.scale_offset[1] + diff_between_scales[1])

        # Обновление предыдущей шкалы
        self.prev_scaled_size = self.scaled_size

        # Масштаб выполнения
        self.map_image.scale_map(scale_factor)

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def next_map_layer(self, reverse=False):
        self.current_map_layer += -1 if reverse else 1
        self.current_map_layer %= len(self.map_layer_names)
        selected = self.map_layer_names[self.current_map_layer]
        self.hud.notification('LayerMap selected: %s' % selected)

    def load_map_layer(self, unload=False):
        selected = self.map_layer_names[self.current_map_layer]
        if unload:
            self.hud.notification('Unloading map layer: %s' % selected)
            self.world.unload_map_layer(selected)
        else:
            self.hud.notification('Loading map layer: %s' % selected)
            self.world.load_map_layer(selected)

    def toggle_radar(self):
        if self.radar_sensor is None:
            self.radar_sensor = RadarSensor(self.player)
        elif self.radar_sensor.sensor is not None:
            self.radar_sensor.sensor.destroy()
            self.radar_sensor = None

    def modify_vehicle_physics(self, actor):
        # Если актер не является транспортным средством, мы не можем использовать физический контроль
        try:
            physics_control = actor.get_physics_control()
            physics_control.use_sweep_wheel_collision = True
            actor.apply_physics_control(physics_control)
        except Exception:
            pass

    def tick(self, clock):
        """Получает актеров для режимов "Герой" и "Карта" и обновляет на их"""
        actors = self.world.get_actors()
        self.hud.time_grab = time.time_ns()
        # Мы также храним трансформации, чтобы при рендеринге не иметь трансформаций предыдущего и текущего тика
        
        self.actors_with_transforms = [(actor, actor.get_transform()) for actor in actors]
        if self.player is not None:
            self.hero_transform = self.player.get_transform()
        
        self.hud.tick(self, clock)

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def map_render(self, display):
        """Прорисовка карты и всех действующих лиц в режиме героя и карты"""
        if self.actors_with_transforms is None:
            return
        self.result_surface.fill(COLOR_BLACK)

        # Разделение актеров по идентификатору типа транспортного средства
        vehicles, traffic_lights, speed_limits, walkers = self._split_actors()

        # Увеличение и уменьшение масштаба
        scale_factor = 1.0 
        self.scaled_size = int(self.map_image.width * scale_factor)
        if self.scaled_size != self.prev_scaled_size:
            self._compute_scale(scale_factor)

        # Рендеринг актеров
        self.actors_surface.fill(COLOR_BLACK)
        self.render_actors(
            self.actors_surface,
            vehicles,
            traffic_lights,
            speed_limits,
            walkers)

        # Показать близлежащих актеров из режима героя
        self._show_nearby_vehicles(vehicles)

        surfaces = ((self.map_image.surface, (0, 0)),
                    (self.actors_surface, (0, 0)),
                    (self.vehicle_id_surface, (0, 0)),
                    )

        angle = 0.0 if self.player is None else self.hero_transform.rotation.yaw + 90.0
        self.traffic_light_surfaces.rotozoom(-angle, self.map_image.scale)

        # Режим героя только в этом использовании
        hero_location_screen = self.map_image.world_to_pixel(self.hero_transform.location)
        hero_front = self.hero_transform.get_forward_vector()
        translation_offset = (hero_location_screen[0] - self.hero_surface.get_width() / 2 + hero_front.x * PIXELS_AHEAD_VEHICLE,
                                (hero_location_screen[1] - self.hero_surface.get_height() / 2 + hero_front.y * PIXELS_AHEAD_VEHICLE))

        clipping_rect = pygame.Rect(translation_offset[0],
                                    translation_offset[1],
                                    self.hero_surface.get_width(),
                                    self.hero_surface.get_height())
        self.clip_surfaces(clipping_rect)

        Util.blits(self.result_surface, surfaces)

        self.border_round_surface.set_clip(clipping_rect)

        self.hero_surface.fill(COLOR_ALUMINIUM_4)
        self.hero_surface.blit(self.result_surface, (-translation_offset[0],
                                                        -translation_offset[1]))

        rotated_result_surface = pygame.transform.rotozoom(self.hero_surface, angle, 0.9).convert()

        if self.camera_manager.index==1:
            center = (display.get_width() / 2, display.get_height() / 2)
            rotation_pivot = rotated_result_surface.get_rect(center=center)
            display.blit(rotated_result_surface, rotation_pivot)

            display.blit(self.border_round_surface, (0, 0))
            self.hud.render(display)
        if self.map_capture and self.hud.speed>0.1:
                    _width = rotated_result_surface.get_width()
                    _height = rotated_result_surface.get_height()
                    _new_width = math.floor(_width / MAP_CAPTURE_CROP_FACTOR)
                    _new_height = math.floor(_height / MAP_CAPTURE_CROP_FACTOR)
                    _new_width_start = math.floor(_width / 2) - math.floor(_new_width / 2)
                    _new_height_start = math.floor(_height / 2) - math.floor(_new_height / 2)
                    slice_rect = pygame.Rect(_new_width_start, _new_height_start, _new_width, _new_height)
                    map_to_save = rotated_result_surface.subsurface(slice_rect)
                    pygame.image.save(map_to_save, MAP_FOLDER+'%06d.png' % self.hud.time_grab)

    def destroy_sensors(self):
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        if self.radar_sensor is not None:
            self.toggle_radar()
        sensors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.imu_sensor.sensor]
        for sensor in sensors:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        if self.player is not None:
            self.player.destroy()


# ==============================================================================
# -- Управление клавиатурой-----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
    """Класс, обрабатывающий ввод с клавиатуры."""
    def __init__(self, world, start_in_autopilot):
        self._autopilot_enabled = start_in_autopilot
        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            self._lights = carla.VehicleLightState.NONE
            world.player.set_autopilot(self._autopilot_enabled)
            world.player.set_light_state(self._lights)
        elif isinstance(world.player, carla.Walker):
            self._control = carla.WalkerControl()
            self._autopilot_enabled = False
            self._rotation = world.player.get_transform().rotation
        else:
            raise NotImplementedError("Actor type not supported")
        self._steer_cache = 0.0
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self, client, world, clock, sync_mode):
        if isinstance(self._control, carla.VehicleControl):
            current_lights = self._lights
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_BACKSPACE:
                    if self._autopilot_enabled:
                        world.player.set_autopilot(False)
                        world.restart()
                        world.player.set_autopilot(True)
                    else:
                        world.restart()
                elif event.key == K_F1:
                    world.hud.toggle_info()
                elif event.key == K_v and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_map_layer(reverse=True)
                elif event.key == K_v:
                    world.next_map_layer()
                elif event.key == K_b and pygame.key.get_mods() & KMOD_SHIFT:
                    world.load_map_layer(unload=True)
                elif event.key == K_b:
                    world.load_map_layer()
                elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
                    world.hud.help.toggle()
                elif event.key == K_TAB:
                    world.camera_manager.toggle_camera()
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_weather(reverse=True)
                elif event.key == K_c:
                    world.next_weather()
                elif event.key == K_g:
                    world.toggle_radar()
                elif event.key == K_BACKQUOTE:
                    world.camera_manager.next_sensor()
                elif event.key == K_n:
                    world.camera_manager.next_sensor()
                elif event.key == K_w and (pygame.key.get_mods() & KMOD_CTRL):
                    if world.constant_velocity_enabled:
                        world.player.disable_constant_velocity()
                        world.constant_velocity_enabled = False
                        world.hud.notification("Disabled Constant Velocity Mode")
                    else:
                        world.player.enable_constant_velocity(carla.Vector3D(17, 0, 0))
                        world.constant_velocity_enabled = True
                        world.hud.notification("Enabled Constant Velocity Mode at 60 km/h")
                elif event.key == K_o:
                    try:
                        if world.doors_are_open:
                            world.hud.notification("Closing Doors")
                            world.doors_are_open = False
                            world.player.close_door(carla.VehicleDoor.All)
                        else:
                            world.hud.notification("Opening doors")
                            world.doors_are_open = True
                            world.player.open_door(carla.VehicleDoor.All)
                    except Exception:
                        pass
                elif event.key == K_t:
                    if world.show_vehicle_telemetry:
                        world.player.show_debug_telemetry(False)
                        world.show_vehicle_telemetry = False
                        world.hud.notification("Disabled Vehicle Telemetry")
                    else:
                        try:
                            world.player.show_debug_telemetry(True)
                            world.show_vehicle_telemetry = True
                            world.hud.notification("Enabled Vehicle Telemetry")
                        except Exception:
                            pass
                elif event.key > K_0 and event.key <= K_9:
                    index_ctrl = 0
                    if pygame.key.get_mods() & KMOD_CTRL:
                        index_ctrl = 9
                    world.camera_manager.set_sensor(event.key - 1 - K_0 + index_ctrl)
                elif event.key == K_r and not (pygame.key.get_mods() & KMOD_CTRL):
                    world.camera_manager.toggle_recording()
                elif event.key == K_r and (pygame.key.get_mods() & KMOD_CTRL):
                    if (world.recording_enabled):
                        client.stop_recorder()
                        world.recording_enabled = False
                        world.hud.notification("Recorder is OFF")
                    else:
                        client.start_recorder("manual_recording.rec")
                        world.recording_enabled = True
                        world.hud.notification("Recorder is ON")
                elif event.key == K_p and (pygame.key.get_mods() & KMOD_CTRL):
                    client.stop_recorder()
                    world.recording_enabled = False
                    # исправление камеры в начале воспроизведения
                    current_index = world.camera_manager.index
                    world.destroy_sensors()
                    # отключить автопилот
                    self._autopilot_enabled = False
                    world.player.set_autopilot(self._autopilot_enabled)
                    world.hud.notification("Replaying file 'manual_recording.rec'")
                    client.replay_file("manual_recording.rec", world.recording_start, 0, 0)
                    world.camera_manager.set_sensor(current_index)
                elif event.key == K_MINUS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start -= 10
                    else:
                        world.recording_start -= 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))
                elif event.key == K_EQUALS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start += 10
                    else:
                        world.recording_start += 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))
                if isinstance(self._control, carla.VehicleControl):
                    if event.key == K_q:
                        self._control.gear = 1 if self._control.reverse else -1
                    elif event.key == K_m:
                        world.map_capture = not world.map_capture
                        world.hud.notification('Map recording %s' %
                                               ('On' if world.map_capture else 'Off'))
                    elif event.key == K_p and not pygame.key.get_mods() & KMOD_CTRL:
                        if not self._autopilot_enabled and not sync_mode:
                            print("WARNING: You are currently in asynchronous mode and could "
                                  "experience some issues with the traffic simulation")
                        self._autopilot_enabled = not self._autopilot_enabled
                        world.player.set_autopilot(self._autopilot_enabled)
                        world.hud.notification(
                            'Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))
                    elif event.key == K_l and pygame.key.get_mods() & KMOD_CTRL:
                        current_lights ^= carla.VehicleLightState.Special1
                    elif event.key == K_l and pygame.key.get_mods() & KMOD_SHIFT:
                        current_lights ^= carla.VehicleLightState.HighBeam
                    elif event.key == K_l:
                        if not self._lights & carla.VehicleLightState.Position:
                            world.hud.notification("Position lights")
                            current_lights |= carla.VehicleLightState.Position
                        else:
                            world.hud.notification("Low beam lights")
                            current_lights |= carla.VehicleLightState.LowBeam
                        if self._lights & carla.VehicleLightState.LowBeam:
                            world.hud.notification("Fog lights")
                            current_lights |= carla.VehicleLightState.Fog
                        if self._lights & carla.VehicleLightState.Fog:
                            world.hud.notification("Lights off")
                            current_lights ^= carla.VehicleLightState.Position
                            current_lights ^= carla.VehicleLightState.LowBeam
                            current_lights ^= carla.VehicleLightState.Fog
                    elif event.key == K_i:
                        current_lights ^= carla.VehicleLightState.Interior
                    elif event.key == K_z:
                        current_lights ^= carla.VehicleLightState.LeftBlinker
                    elif event.key == K_x:
                        current_lights ^= carla.VehicleLightState.RightBlinker

        if not self._autopilot_enabled:
            if isinstance(self._control, carla.VehicleControl):
                self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
                self._control.reverse = self._control.gear < 0
                # Установка световых приборов автомобиля с автоматическим управлением
                if self._control.brake:
                    current_lights |= carla.VehicleLightState.Brake
                else: # Удалить флаг тормоза
                    current_lights &= ~carla.VehicleLightState.Brake
                if self._control.reverse:
                    current_lights |= carla.VehicleLightState.Reverse
                else: # Убрать флаг заднего хода
                    current_lights &= ~carla.VehicleLightState.Reverse
                if current_lights != self._lights: # Изменение состояния освещения только в случае необходимости
                    self._lights = current_lights
                    world.player.set_light_state(carla.VehicleLightState(self._lights))
            elif isinstance(self._control, carla.WalkerControl):
                self._parse_walker_keys(pygame.key.get_pressed(), clock.get_time(), world)
            world.player.apply_control(self._control)

    def _parse_vehicle_keys(self, keys, milliseconds):
        if keys[K_UP] or keys[K_w]:
            self._control.throttle = min(self._control.throttle + 0.01, 1.00)
        else:
            self._control.throttle = 0.0

        if keys[K_DOWN] or keys[K_s]:
            self._control.brake = min(self._control.brake + 0.2, 1)
        else:
            self._control.brake = 0

        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            if self._steer_cache > 0:
                self._steer_cache = 0
            else:
                self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            if self._steer_cache < 0:
                self._steer_cache = 0
            else:
                self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.hand_brake = keys[K_SPACE]

    def _parse_walker_keys(self, keys, milliseconds, world):
        self._control.speed = 0.0
        if keys[K_DOWN] or keys[K_s]:
            self._control.speed = 0.0
        if keys[K_LEFT] or keys[K_a]:
            self._control.speed = .01
            self._rotation.yaw -= 0.08 * milliseconds
        if keys[K_RIGHT] or keys[K_d]:
            self._control.speed = .01
            self._rotation.yaw += 0.08 * milliseconds
        if keys[K_UP] or keys[K_w]:
            self._control.speed = world.player_max_speed_fast if pygame.key.get_mods() & KMOD_SHIFT else world.player_max_speed
        self._control.jump = keys[K_SPACE]
        self._rotation.yaw = round(self._rotation.yaw, 1)
        self._control.direction = self._rotation.get_forward_vector()

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 16), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()
        self.time_grab = None
        self.speed = None

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        t = world.player.get_transform()
        v = world.player.get_velocity()
        self.speed = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
        c = world.player.get_control()
        compass = world.imu_sensor.compass
        heading = 'N' if compass > 270.5 or compass < 89.5 else ''
        heading += 'S' if 90.5 < compass < 269.5 else ''
        heading += 'E' if 0.5 < compass < 179.5 else ''
        heading += 'W' if 180.5 < compass < 359.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.map.name.split('/')[-1],
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (self.speed),
            u'Compass:% 17.0f\N{DEGREE SIGN} % 2s' % (compass, heading),
            'Accelero: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.accelerometer),
            'Gyroscop: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.gyroscope),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % t.location.z,
            '']
        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake)]
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [
                ('Speed:', c.speed, 0.0, 5.556),
                ('Jump:', c.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']
            distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.player.id]
            for d, vehicle in sorted(vehicles, key=lambda vehicles: vehicles[0]):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- Текст справки ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    """Класс-помощник для обработки вывода текста с помощью pygame"""
    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.line_space = 18
        self.dim = (780, len(lines) * self.line_space + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * self.line_space))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)


# ==============================================================================
# -- Датчик столкновения -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)


# ==============================================================================
# -- Датчик движения по полосе --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None

        # Если объект захвата не является автомобилем, мы не можем использовать датчик вторжения в полосу движения
        if parent_actor.type_id.startswith("vehicle."):
            self._parent = parent_actor
            self.hud = hud
            world = self._parent.get_world()
            bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
            self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))


# ==============================================================================
# -- Gnss ----------------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


# ==============================================================================
# -- Гиростабилизатор -----------------------------------------------------------------
# ==============================================================================


class IMUSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.accelerometer = (0.0, 0.0, 0.0)
        self.gyroscope = (0.0, 0.0, 0.0)
        self.compass = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.imu')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda sensor_data: IMUSensor._IMU_callback(weak_self, sensor_data))

    @staticmethod
    def _IMU_callback(weak_self, sensor_data):
        self = weak_self()
        if not self:
            return
        limits = (-99.9, 99.9)
        self.accelerometer = (
            max(limits[0], min(limits[1], sensor_data.accelerometer.x)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.y)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.z)))
        self.gyroscope = (
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.x))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.y))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.z))))
        self.compass = math.degrees(sensor_data.compass)


# ==============================================================================
# -- Радар ---------------------------------------------------------------
# ==============================================================================


class RadarSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        bound_x = 0.5 + self._parent.bounding_box.extent.x
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        bound_z = 0.5 + self._parent.bounding_box.extent.z

        self.velocity_range = 7.5 # м/с
        world = self._parent.get_world()
        self.debug = world.debug
        bp = world.get_blueprint_library().find('sensor.other.radar')
        bp.set_attribute('horizontal_fov', str(35))
        bp.set_attribute('vertical_fov', str(20))
        self.sensor = world.spawn_actor(
            bp,
            carla.Transform(
                carla.Location(x=bound_x + 0.05, z=bound_z+0.05),
                carla.Rotation(pitch=5)),
            attach_to=self._parent)
        
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda radar_data: RadarSensor._Radar_callback(weak_self, radar_data))

    @staticmethod
    def _Radar_callback(weak_self, radar_data):
        self = weak_self()
        if not self:
            return
        

        current_rot = radar_data.transform.rotation
        for detect in radar_data:
            azi = math.degrees(detect.azimuth)
            alt = math.degrees(detect.altitude)
            # 0,25 немного корректирует расстояние, чтобы точки были хорошо видны.
            fw_vec = carla.Vector3D(x=detect.depth - 0.25)
            carla.Transform(
                carla.Location(),
                carla.Rotation(
                    pitch=current_rot.pitch + alt,
                    yaw=current_rot.yaw + azi,
                    roll=current_rot.roll)).transform(fw_vec)

            def clamp(min_v, max_v, value):
                return max(min_v, min(value, max_v))

            norm_velocity = detect.velocity / self.velocity_range # [-1, 1]
            r = int(clamp(0.0, 1.0, 1.0 - norm_velocity) * 255.0)
            g = int(clamp(0.0, 1.0, 1.0 - abs(norm_velocity)) * 255.0)
            b = int(abs(clamp(- 1.0, 0.0, - 1.0 - norm_velocity)) * 255.0)
            self.debug.draw_point(
                radar_data.transform.location + fw_vec,
                size=0.075,
                life_time=0.06,
                persistent_lines=False,
                color=carla.Color(r, g, b))

# ==============================================================================
# -- Диспетчер камер -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, parent_actor, hud, gamma_correction):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        Attachment = carla.AttachmentType

        if not self._parent.type_id.startswith("walker.pedestrian"):
            self._camera_transforms = [
                (carla.Transform(carla.Location(x=CAMERA_POS_X, y=+0.0, z=CAMERA_POS_Z), carla.Rotation(pitch=+0.0)), Attachment.Rigid)]
        else:
            self._camera_transforms = [
                (carla.Transform(carla.Location(x=-2.5, z=0.0), carla.Rotation(pitch=-8.0)), Attachment.SpringArm),
                (carla.Transform(carla.Location(x=1.6, z=1.7)), Attachment.Rigid),
                (carla.Transform(carla.Location(x=2.5, y=0.5, z=0.0), carla.Rotation(pitch=-8.0)), Attachment.SpringArm),
                (carla.Transform(carla.Location(x=-4.0, z=2.0), carla.Rotation(pitch=6.0)), Attachment.SpringArm),
                (carla.Transform(carla.Location(x=0, y=-2.5, z=-0.0), carla.Rotation(yaw=90.0)), Attachment.Rigid)]

        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)', {}],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)', {}],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)', {}],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)', {}],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette, 'Camera Semantic Segmentation (CityScapes Palette)', {}],
            ['sensor.camera.instance_segmentation', cc.CityScapesPalette, 'Camera Instance Segmentation (CityScapes Palette)', {}],
            ['sensor.camera.instance_segmentation', cc.Raw, 'Camera Instance Segmentation (Raw)', {}],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)', {'range': '50'}],
            ['sensor.camera.dvs', cc.Raw, 'Dynamic Vision Sensor', {}],
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB Distorted',
                {'lens_circle_multiplier': '3.0',
                'lens_circle_falloff': '3.0',
                'chromatic_aberration_intensity': '0.5',
                'chromatic_aberration_offset': '0'}],
            ['sensor.camera.optical_flow', cc.Raw, 'Optical Flow', {}],
        ]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
                if bp.has_attribute('gamma'):
                    bp.set_attribute('gamma', str(gamma_correction))
                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
            elif item[0].startswith('sensor.lidar'):
                self.lidar_range = 50

                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
                    if attr_name == 'range':
                        self.lidar_range = float(attr_value)

            item.append(bp)
        self.index = None

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else \
            (force_respawn or (self.sensors[index][2] != self.sensors[self.index][2]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])
            
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / (2.0 * self.lidar_range)
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        elif self.sensors[self.index][0].startswith('sensor.camera.dvs'):
            # Пример преобразования raw_data из датчика carla.DVSEventArray в массив NumPy
            # и использования его в качестве изображения
            dvs_events = np.frombuffer(image.raw_data, dtype=np.dtype([
                ('x', np.uint16), ('y', np.uint16), ('t', np.int64), ('pol', np.bool)]))
            dvs_img = np.zeros((image.height, image.width, 3), dtype=np.uint8)
            # Синий - положительный, красный - отрицательный
            dvs_img[dvs_events[:]['y'], dvs_events[:]['x'], dvs_events[:]['pol'] * 2] = 255
            self.surface = pygame.surfarray.make_surface(dvs_img.swapaxes(0, 1))
        elif self.sensors[self.index][0].startswith('sensor.camera.optical_flow'):
            image = image.get_color_coded_flow()
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording and self.hud.speed>0.1:
            image.save_to_disk(IMG_FOLDER+'%08d' % self.hud.time_grab)
            
# ==============================================================================
# -- игровой цикл() ---------------------------------------------------------------
# ==============================================================================


def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None
    original_settings = None

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(20.0)
        sim_world = client.load_world(args.map) 

        if args.sync:
            original_settings = sim_world.get_settings()
            settings = sim_world.get_settings()
            if not settings.synchronous_mode:
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
            sim_world.apply_settings(settings)

            traffic_manager = client.get_trafficmanager()
            traffic_manager.set_synchronous_mode(True)

        if args.autopilot and not sim_world.get_settings().synchronous_mode:
            print("WARNING: You are currently in asynchronous mode and could "
                  "experience some issues with the traffic simulation")
        # Это основное отображение из исходного примера manual_control.py
        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        display.fill((0,0,0))
        pygame.display.flip()

        hud = HUD(args.width, args.height)
        world = World(sim_world, hud, args)
        controller = KeyboardControl(world, args.autopilot)

        if args.sync:
            sim_world.tick()
        else:
            sim_world.wait_for_tick()

        clock = pygame.time.Clock()
        while True:
            if args.sync:
                sim_world.tick()
            clock.tick_busy_loop(60)
            if controller.parse_events(client, world, clock, args.sync):
                return
            world.tick(clock)
            # Переписывание 2-го датчика при просмотре карты: когда вы выбираете 2 датчика - просмотр карты переписывает его
            world.render(display)
            world.map_render(display) # это отображается на главном дисплее только при выборе второго датчика
            pygame.display.flip()

    finally:

        if original_settings:
            sim_world.apply_settings(original_settings)

        if (world and world.recording_enabled):
            client.stop_recorder()

        if world is not None:
            world.destroy()

        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        default= True,
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default=f"{CAM_WIDTH}x{CAM_HEIGHT}",
        help=f'window resolution (default: {CAM_WIDTH}x{CAM_HEIGHT})')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--generation',
        metavar='G',
        default='2',
        help='restrict to certain actor generation (values: "1","2","All" - default: "2")')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    argparser.add_argument(
        '--show-triggers',
        action='store_true',
        help='show trigger boxes of traffic signs')
    argparser.add_argument(
        '--show-connections',
        action='store_true',
        help='show waypoint connections')
    argparser.add_argument(
        '--show-spawn-points',
        action='store_true',
        help='show recommended spawn points')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Activate synchronous mode execution')
    argparser.add_argument(
        '--map',
        metavar='TOWN',
        default='Town05',
        help='start a new episode at the given TOWN')
    argparser.add_argument(
        '--clear',
        action='store_true',
        default=True,
        help='Clean all vehicles and sensors from sim beforehand')
    argparser.add_argument(
        '--map-capture',
        action='store_true',
        default=False,
        help='Clean all vehicles and sensors from sim beforehand')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:

        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()
