//=============================================================================================
// Program: Parametrikus görbék
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!!
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Sánta Dániel
// Neptun : OGLSI2
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char * const vertexSource = R"(
	#version 330
	precision highp float;		// normal floats, makes no difference on desktop computers

	uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix
	layout(location = 0) in vec2 vp;	// Varying input: vp = vertex position is expected in attrib array 0

	void main() {
		gl_Position = vec4(vp.x, vp.y, 0, 1) * MVP;		// transform vp from modeling space to normalized device space
	}
)";

// fragment shader in GLSL
const char * const fragmentSource = R"(
	#version 330
	precision highp float;	// normal floats, makes no difference on desktop computers

	uniform vec3 color;		// uniform variable, the color of the primitive
	out vec4 outColor;		// computed color of the current pixel

	void main() {
		outColor = vec4(color, 1);	// computed color is the color of the primitive
	}
)";

GPUProgram gpuProgram; // vertex and fragment shaders

mat4 M(1, 0, 0, 0,
         0, 1, 0, 0,
         0, 0, 1, 0,
         0, 0, 0, 1);
mat4 MVP;

class Camera {
    vec2 wCenter = vec2(0, 0);// center in world coords
    vec2 wSize = vec2(30,30); // width and height in world coords
public:
    mat4 V() { return TranslateMatrix(-wCenter); }
    mat4 P() { // projection matrix
        return ScaleMatrix(vec2(2/wSize.x, 2/wSize.y));
    }
    mat4 Vinv() { // inverse view matrix
        return TranslateMatrix(wCenter);
    }
    mat4 Pinv() { // inverse projection matrix
        return ScaleMatrix(vec2(wSize.x/2, wSize.y/2));
    }
    void Zoom(float s) { wSize = wSize * s; }
    void Pan(vec2 t) { wCenter = wCenter + t; }
};

Camera camera;

void calcMVP()
{
    MVP = M * camera.V() * camera.P();
}

vec2 screenToWorld(const vec2& pixelCoord, Camera camera)
{
    vec4 viewport = vec4(pixelCoord.x, pixelCoord.y, 1, 1);
    vec4 cam = viewport * camera.Pinv();
    vec4 world = cam * camera.Vinv();

    return vec2(world.x, world.y) / world.w;
}

enum State
{
    Lagrange, Bezier, CatmullRom
};

const int nTessVertices = 100;

class Curve
{
private:
    unsigned int vao, vbo; // GPU
public:
    std::vector<vec2> cps;

    virtual void AddControlPoints(vec2 p) {};

    vec2 *pickPoint(vec2 pp, float threshold)
    {
        for (auto& p : cps)
            if (length(pp - p) < threshold)
                return &p;
        return nullptr;
    }

    bool pickPointSame(vec2 pp)
    {
        if (cps.size() != 0)
            if (length(pp - cps.back()) == 0)
                return true;
        return false;
    }

    void init()
    {
        glGenVertexArrays(1, &vao);	// get 1 vao id
        glBindVertexArray(vao);
        glGenBuffers(1, &vbo);	// Generate 1 buffer
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glEnableVertexAttribArray(0);  // AttribArray 0
        glVertexAttribPointer(0,       // vbo -> AttribArray 0
                              2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
                              0, NULL); 		     // stride, offset: tightly packed
        calcMVP();
    }

    void updateGPU()
    {
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER,vbo);
        glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
                     cps.size() * sizeof(vec2),  // # bytes
                     &cps[0],	      	// address
                     GL_STATIC_DRAW);	// we do not change later
    }

    virtual std::vector<vec2> GenVertexData() { return std::vector<vec2>(); }

    void draw()
    {
        calcMVP();
        gpuProgram.setUniform(MVP, "MVP");

        gpuProgram.setUniform(vec3(1,1,0), "color");
        std::vector<vec2> curvePoints = GenVertexData();
        glBindVertexArray(vao);  // Draw call
        glBindBuffer(GL_ARRAY_BUFFER, vbo); // Frissítjük a VBO-t
        glBufferData(GL_ARRAY_BUFFER, curvePoints.size() * sizeof(vec2), &curvePoints[0], GL_STATIC_DRAW);
        glDrawArrays(GL_LINE_STRIP, 0, curvePoints.size());

        glBindVertexArray(vao);  // Draw call
        gpuProgram.setUniform(vec3(1,0,0), "color");
        updateGPU();
        glDrawArrays(GL_POINTS, 0, cps.size());
    }
};

class BezierCurve : public Curve
{
    float B(unsigned int i, float t) {
        float choose = 1;
        for(unsigned int j = 1; j <= i; j++)
            choose *= (float)(cps.size()-j)/j;
        return choose * pow(t, i) * pow(1-t, cps.size()-1-i);
    }
public:
    void AddControlPoint(vec2 cp) { cps.push_back(cp); updateGPU(); }

    vec2 r(float t) {
        vec2 rt(0, 0);
        for(unsigned int i = 0; i < cps.size(); i++)
            rt = rt + cps[i] * B(i,t);
        return rt;
    }

    std::vector<vec2> GenVertexData() {
        std::vector<vec2> vertices;
        for(unsigned int i = 0; i <= nTessVertices; ++i) {
            float t = (float)i / nTessVertices;
            vertices.push_back(r(t));
        }
        return vertices;
    }

    void deleteCurve()
    {
        cps.clear();
    }
};

class LagrangeCurve : public Curve
{
    std::vector<float> ts;
    float L(unsigned int i, float t) {
        float Li = 1.0f;
        for(unsigned int j = 0; j < cps.size(); j++)
            if (j != i)
                Li *= (t - ts[j])/(ts[i] - ts[j]);
        return Li;
    }

    void updateTs()
    {
        ts.clear();
        float ti = 0.0f;
        ts.push_back(ti);
        if (cps.size() == 1)
            return;

        for (unsigned int i = 0; i < cps.size() - 1; ++i)
            ts.push_back(ts.at(i) + length(cps.at(i) - cps.at(i + 1)));

        float dist = ts.at(cps.size() - 1);

        for (unsigned int i = 0; i < ts.size(); ++i)
            ts.at(i) = ts.at(i) / dist;
    }

public:
    void AddControlPoint(vec2 cp)
    {
        cps.push_back(cp);
        updateTs();
        updateGPU();
    }

    vec2 r(float t) {
        vec2 rt(0, 0);
        for(unsigned int i = 0; i < cps.size(); i++)
            rt = rt + cps[i] * L(i,t);
        return rt;
    }

    std::vector<vec2> GenVertexData()
    {
        std::vector<vec2> vertices;
        for(unsigned int i = 0; i <= nTessVertices; ++i) {
            float t = (float)i / nTessVertices;
            vertices.push_back(r(t));
        }
        return vertices;
    }

    void deleteCurve()
    {
        cps.clear();
        ts.clear();
    }
};

class CatmullRomCurve : public Curve
{
    public: std::vector<float> ts;
    float tenzio = 0;

    vec2 Hermite(vec2 p0, vec2 v0, float t0, vec2 p1, vec2 v1, float t1, float t)
    {
        vec2 rt;
        vec2 a0 = p0;
        vec2 a1 = v0;
        vec2 a2 =   (3 * (p1 - p0) / pow(t1 - t0, 2)) -
                    ((v1 + 2 * v0) / (t1 - t0));
        vec2 a3 =   (2 * (p0 - p1) / pow(t1 - t0, 3)) +
                    ((v1 + v0) / (pow(t1 - t0, 2)));
        rt = a3 * pow(t - t0, 3) + a2 * pow(t - t0, 2) + a1 * (t - t0) + a0;

        printf("a1:%f %f",a1.x, a1.y); printf("\n");
        printf("a2:%f %f",a2.x, a2.y); printf("\n");
        printf("a3:%f %f",a3.x, a3.y); printf("\n");

        return rt;
    }

    void updateTs()
    {
        ts.clear();
        float ti = 0.0f;
        ts.push_back(ti);
        if (cps.size() == 1)
            return;

        for (unsigned int i = 0; i < cps.size() - 1; ++i)
            ts.push_back(ts.at(i) + length(cps.at(i) - cps.at(i + 1)));

        float dist = ts.at(cps.size() - 1);

        for (unsigned int i = 0; i < ts.size(); ++i)
            ts.at(i) = ts.at(i) / dist;
    }

public:
    void AddControlPoint(vec2 cp)
    {
        cps.push_back(cp);
        updateTs();
        updateGPU();
    }

    void incrementTenzio()
    {
        tenzio += 0.1;
    }

    void decrementTenzio()
    {
        tenzio -= 0.1;
    }

    vec2 r(float t)
    {
        if (cps.size() == 2)
            return Hermite(cps[0], 0, ts[0], cps[1], 0, ts[1], t);
        if (cps.size() != 0)
            for(unsigned int i = 0; i < cps.size() - 1; i++)
                if (ts[i] <= t && t <= ts[i+1])
                {
                    vec2 v0, v1;
                    if (i == 0)
                    {
                        v0 = vec2(0,0);
                        v1 =    (((1-tenzio) / 2) *
                                (((cps[i+2] - cps[i+1]) / (ts[i+2] - ts[i+1]))+
                                ((cps[i+1] - cps[i]) / (ts[i+1] - ts[i]))));
                    }
                    if (i == cps.size() - 2)
                    {
                        v0 =    (((1-tenzio) / 2) *
                                 (((cps[i+1] - cps[i]) / (ts[i+1] - ts[i]))+
                                  ((cps[i] - cps[i-1]) / (ts[i] - ts[i-1]))));
                        v1 = vec2(0,0);
                    }
                    if (i != 0 && i != cps.size() - 2)
                    {
                        v0 =    (((1-tenzio) / 2) *
                                 (((cps[i+1] - cps[i]) / (ts[i+1] - ts[i]))+
                                  ((cps[i] - cps[i-1]) / (ts[i] - ts[i-1]))));
                        v1 =    (((1-tenzio) / 2) *
                                 (((cps[i+2] - cps[i+1]) / (ts[i+2] - ts[i+1]))+
                                  ((cps[i+1] - cps[i]) / (ts[i+1] - ts[i]))));
                    }
                    printf("cps(i):%f %f",cps[i].x, cps[i].y); printf("\n");
                    printf("v0:%f %f",v0.x, v0.y); printf("\n");
                    printf("v1:%f %f",v1.x, v1.y); printf("\n");
                    //printf("v0:%f %f",v0.x, v0.y); printf("\n");
                    return Hermite(cps[i], v0, ts[i],cps[i+1], v1, ts[i+1], t);
                }
        return vec2(0,0);
    }

    std::vector<vec2> GenVertexData()
    {
        std::vector<vec2> vertices;
        for(unsigned int i = 0; i <= nTessVertices; ++i) {
            float t = (float)i / nTessVertices;
            vertices.push_back(r(t));
        }
        return vertices;
    }

    void deleteCurve()
    {
        cps.clear();
        ts.clear();
    }
};

bool isDrag = false;

BezierCurve b;
LagrangeCurve l;
CatmullRomCurve c;
State curveState = Lagrange;

// Initialization, create an OpenGL context
void onInitialization() {
    glViewport(0, 0, windowWidth, windowHeight);

    b.init();
    l.init();
    c.init();
    // create program for the GPU
    gpuProgram.create(vertexSource, fragmentSource, "outColor");


    c.cps.push_back(vec2(1,1));
    c.cps.push_back(vec2(2,4));
    c.cps.push_back(vec2(3,9));
    c.cps.push_back(vec2(4,16));
    c.cps.push_back(vec2(5,25));

    c.ts.push_back(1);
    c.ts.push_back(2);
    c.ts.push_back(3);
    c.ts.push_back(4);
    c.ts.push_back(5);

    printf("%f %f", c.r(2.5).x, c.r(2.5).y);
}

// Window has become invalid: Redraw
void onDisplay() {
    glClearColor(0, 0, 0, 0);     // background color
    glClear(GL_COLOR_BUFFER_BIT); // clear frame buffer

    int location = glGetUniformLocation(gpuProgram.getId(), "MVP");	// Get the GPU location of uniform variable MVP
    const float* MVPData = reinterpret_cast<const float*>(&MVP);
    glUniformMatrix4fv(location, 1, GL_TRUE, MVPData);	// Load a 4x4 row-major float matrix to the specified location

    glPointSize(10);
    glLineWidth(2);
    switch (curveState)
    {
        case Bezier:
            b.draw();
        break;
        case Lagrange:
            l.draw();
        break;
        case CatmullRom:
            c.draw();
        break;
    }
    glutSwapBuffers(); // exchange buffers for double buffering
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
    if (key == 'z')
    {
        camera.Zoom(1/1.1);
    }
    if (key == 'Z')
    {
        camera.Zoom(1.1);
    }
    if (key == 'P')
    {
        camera.Pan(vec2(1,0));
    }
    if (key == 'p')
    {
        camera.Pan(vec2(-1,0));
    }
    if (key == 'b')
    {
        curveState = Bezier;
        l.deleteCurve();
        l.updateGPU();
        c.deleteCurve();
        c.updateGPU();
    }
    if (key == 'l')
    {
        curveState = Lagrange;
        b.deleteCurve();
        b.updateGPU();
        c.deleteCurve();
        c.updateGPU();
    }
    if (key == 'c')
    {
        curveState = CatmullRom;
        l.deleteCurve();
        l.updateGPU();
        b.deleteCurve();
        b.updateGPU();
    }
    if (key == 't')
    {
        c.decrementTenzio();
        c.updateGPU();
    }
    if (key == 'T')
    {
        c.incrementTenzio();
        c.updateGPU();
    }

    glutPostRedisplay();
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
    // Convert to normalized device space
    float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
    float cY = 1.0f - 2.0f * pY / windowHeight;

    if (isDrag)
    {
        vec2 pp = screenToWorld(vec2(cX, cY), camera);
        vec2* picked;
        switch (curveState)
        {
            case Bezier:
                picked = b.pickPoint(pp, 0.1);
                break;
            case Lagrange:
                picked = l.pickPoint(pp, 0.1);
                break;
            case CatmullRom:
                picked = c.pickPoint(pp, 0.1);
                break;
        }
        if (picked != nullptr)
        {
            picked->x = pp.x;
            picked->y = pp.y;
            switch (curveState)
            {
                case Bezier:
                    b.updateGPU();
                    b.GenVertexData();
                    break;
                case Lagrange:
                    l.updateGPU();
                    l.GenVertexData();
                    break;
                case CatmullRom:
                    c.updateGPU();
                    c.GenVertexData();
                    break;
            }
            c.updateGPU();
            c.GenVertexData();
            glutPostRedisplay();
        }
    }
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
    // Convert to normalized device space
    float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
    float cY = 1.0f - 2.0f * pY / windowHeight;

    vec2 pp = screenToWorld(vec2(cX, cY), camera);

    if (state == GLUT_DOWN && button == GLUT_LEFT_BUTTON)
    {
        bool isSame;
        switch (curveState)
        {
            case Bezier:
                isSame = b.pickPointSame(pp);
                break;
            case Lagrange:
                isSame = l.pickPointSame(pp);
                break;
            case CatmullRom:
                isSame = c.pickPointSame(pp);
                break;
        }
        if (!isSame)
        {
            switch (curveState)
            {
                case Bezier:
                    b.AddControlPoint(pp);
                    b.GenVertexData();
                    break;
                case Lagrange:
                    l.AddControlPoint(pp);
                    l.GenVertexData();
                    break;
                case CatmullRom:
                    c.AddControlPoint(pp);
                    c.GenVertexData();
                    break;
            }
        }
    }
    if (state == GLUT_DOWN && button == GLUT_RIGHT_BUTTON)
    {
        isDrag = true;
    }
    else
    {
        isDrag = false;
    }
}

void onIdle() {
}